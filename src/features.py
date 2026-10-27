"""
src/features.py

Context-aware, strictly causal feature engineering for short-term appliance
energy forecasting. Implements cyclical temporal encodings, indoor/outdoor
thermodynamic gradients, activity-zone differentials, and leak-free lags/rollings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_processing import UCIDatasetSchema

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """
    Configuration for causal feature generation.
    Note: 1 step = 10 minutes.
      - 6 steps   = 1 hour
      - 18 steps  = 3 hours
      - 36 steps  = 6 hours
      - 144 steps = 24 hours (1 day)
    """

    forecast_horizon: int = 1  # Predict t using data available up to t - forecast_horizon
    target_lags: Tuple[int, ...] = (1, 2, 3, 6, 12, 18, 36, 144)
    rolling_windows: Tuple[int, ...] = (6, 18, 36, 144)
    ema_spans: Tuple[int, ...] = (6, 18, 72)
    sensor_diff_steps: Tuple[int, ...] = (1, 6)  # 10-min and 1-hour thermal rate of change
    drop_contemporaneous_lights: bool = True
    dropna_after_engineering: bool = True


class EnergyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that builds temporal, thermodynamic,
    and autoregressive features without look-ahead bias.
    """

    def __init__(
        self,
        schema: Optional[UCIDatasetSchema] = None,
        config: Optional[FeatureConfig] = None,
    ) -> None:
        self.schema = schema or UCIDatasetSchema()
        self.config = config or FeatureConfig()
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "EnergyFeatureEngineer":
        """Stateless causal transformation; validates index type on fit."""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must have a DatetimeIndex.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the complete engineered feature matrix.

        Args:
            X: Cleaned DataFrame indexed by a 10-minute DatetimeIndex.

        Returns:
            DataFrame containing the target column (`Appliances`) and leak-free features.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must have a DatetimeIndex.")

        df = X.copy()
        horizon = self.config.forecast_horizon
        if horizon <= 0:
            raise ValueError("forecast_horizon must be at least 1 step to prevent target leakage.")

        logger.info("Engineering features with forecast_horizon=%d step(s).", horizon)

        # 1. Calendar, Cyclical & Time-of-Day Regime Features
        df = self._add_temporal_features(df)

        # 2. Spatial & Thermodynamic Sensor Aggregations (shifted by horizon if forecasting ahead)
        df = self._add_thermodynamic_features(df)

        # 3. Strictly Causal Target & Auxiliary Load Lag Features
        df = self._add_lag_features(df, horizon=horizon)

        # 4. Strictly Causal Rolling & Exponential Moving Average Features
        df = self._add_rolling_features(df, horizon=horizon)

        # 5. Sensor Rate-of-Change (Momentum) Features
        df = self._add_sensor_momentum_features(df)

        # 6. Drop contemporaneous 'lights' so only lagged 'lights' remain as predictors
        if (
            self.config.drop_contemporaneous_lights
            and self.schema.auxiliary_load_col in df.columns
        ):
            df = df.drop(columns=[self.schema.auxiliary_load_col])

        # 7. Drop initial warm-up rows induced by max lag / rolling window
        if self.config.dropna_after_engineering:
            initial_len = len(df)
            df = df.dropna(how="any")
            logger.info(
                "Dropped %d warm-up rows from lagging/rolling. Active rows: %d",
                initial_len - len(df),
                len(df),
            )

        self.feature_names_out_ = [
            col for col in df.columns if col != self.schema.target_col
        ]
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates calendar indicators, cyclical encodings, and peak-hour regime flags."""
        idx = df.index
        hour_float = idx.hour + (idx.minute / 60.0)
        day_of_week = idx.dayofweek

        df["hour"] = idx.hour
        df["minute"] = idx.minute
        df["day_of_week"] = day_of_week
        df["is_weekend"] = (day_of_week >= 5).astype(np.int8)

        # Cyclical encodings preserve continuity across midnight (23:50 -> 00:00)
        df["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
        df["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

        # Residential occupancy & activity regimes (Morning: 07-10, Evening Peak: 17-21)
        df["is_morning_ramp"] = ((idx.hour >= 7) & (idx.hour <= 10)).astype(np.int8)
        df["is_evening_peak_window"] = ((idx.hour >= 17) & (idx.hour <= 21)).astype(np.int8)
        df["is_night_baseload"] = ((idx.hour >= 0) & (idx.hour <= 5)).astype(np.int8)

        return df

    def _add_thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes indoor spatial aggregations, indoor-outdoor thermal gradients,
        and high-load room differentials (Kitchen T1 & Laundry T3 vs house baseline).
        """
        in_temp = [c for c in self.schema.indoor_temp_cols if c in df.columns]
        in_rh = [c for c in self.schema.indoor_humidity_cols if c in df.columns]

        # Whole-house indoor climate summaries (excluding T6/RH_6 which are outdoor North)
        df["T_indoor_mean"] = df[in_temp].mean(axis=1)
        df["T_indoor_std"] = df[in_temp].std(axis=1)
        df["T_indoor_range"] = df[in_temp].max(axis=1) - df[in_temp].min(axis=1)

        df["RH_indoor_mean"] = df[in_rh].mean(axis=1)
        df["RH_indoor_std"] = df[in_rh].std(axis=1)

        # Indoor vs. Outdoor thermal & humidity gradients (drives HVAC & heat loss)
        if "T_out" in df.columns:
            df["temp_gradient_station"] = df["T_indoor_mean"] - df["T_out"]
            df["dewpoint_depression"] = df["T_out"] - df["Tdewpoint"]
        if "T6" in df.columns:
            df["temp_gradient_envelope"] = df["T_indoor_mean"] - df["T6"]
        if "RH_out" in df.columns:
            df["rh_gradient_station"] = df["RH_indoor_mean"] - df["RH_out"]

        # Active appliance room anomalies: Kitchen (T1, RH_1) & Laundry (T3, RH_3)
        # Cooking and washing/drying generate localized heat & humidity spikes
        if "T1" in df.columns:
            df["kitchen_temp_anomaly"] = df["T1"] - df["T_indoor_mean"]
        if "RH_1" in df.columns:
            df["kitchen_rh_anomaly"] = df["RH_1"] - df["RH_indoor_mean"]
        if "T3" in df.columns:
            df["laundry_temp_anomaly"] = df["T3"] - df["T_indoor_mean"]
        if "RH_3" in df.columns:
            df["laundry_rh_anomaly"] = df["RH_3"] - df["RH_indoor_mean"]

        return df

    def _add_lag_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Adds strictly causal lagged features for `Appliances` and `lights`.
        A configured lag `k` is shifted by `max(k, horizon)` to guarantee zero leakage.
        """
        target = self.schema.target_col
        lights = self.schema.auxiliary_load_col

        new_cols = {}
        for lag in self.config.target_lags:
            effective_lag = max(lag, horizon)
            col_name = f"{target}_lag_{effective_lag}"
            if col_name not in new_cols:
                new_cols[col_name] = df[target].shift(effective_lag)

            if lights in df.columns and effective_lag in (horizon, 6, 144):
                lights_col = f"{lights}_lag_{effective_lag}"
                new_cols[lights_col] = df[lights].shift(effective_lag)

        # Explicit short-term load velocity (change between t - horizon and t - horizon - 1)
        base_lag = df[target].shift(horizon)
        prev_lag = df[target].shift(horizon + 1)
        prev_hour_lag = df[target].shift(horizon + 6)

        new_cols[f"{target}_diff_10m"] = base_lag - prev_lag
        new_cols[f"{target}_diff_60m"] = base_lag - prev_hour_lag
        new_cols[f"{target}_ratio_to_24h_lag"] = base_lag / (
            df[target].shift(max(144, horizon)) + 1.0
        )

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_rolling_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Computes rolling window aggregations on the target shifted by `horizon`.
        Shifting BEFORE `.rolling()` guarantees the current target is never included.
        """
        target = self.schema.target_col
        shifted_target = df[target].shift(horizon)

        new_cols = {}
        for window in self.config.rolling_windows:
            roll = shifted_target.rolling(window=window, min_periods=window)
            new_cols[f"{target}_roll_mean_{window}"] = roll.mean()
            new_cols[f"{target}_roll_std_{window}"] = roll.std()
            new_cols[f"{target}_roll_min_{window}"] = roll.min()
            new_cols[f"{target}_roll_max_{window}"] = roll.max()
            new_cols[f"{target}_roll_skew_{window}"] = roll.skew()

        for span in self.config.ema_spans:
            new_cols[f"{target}_ema_{span}"] = shifted_target.ewm(
                span=span, adjust=False
            ).mean()

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_sensor_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Captures short-term temperature and humidity trajectories (10-min and 60-min
        deltas) in key zones to detect active appliance heat-up/cool-down cycles.
        """
        momentum_sensors = ["T1", "T2", "T3", "RH_1", "RH_3", "T_out", "T_indoor_mean"]
        available_sensors = [s for s in momentum_sensors if s in df.columns]

        new_cols = {}
        for sensor in available_sensors:
            for step in self.config.sensor_diff_steps:
                new_cols[f"{sensor}_delta_{step}"] = df[sensor] - df[sensor].shift(step)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)