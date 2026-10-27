"""
src/data_processing.py

Data ingestion, schema validation, chronological grid alignment, and
causal imputation for the UCI Appliances Energy Prediction dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@dataclass(frozen=True)
class UCIDatasetSchema:
    """Canonical schema definition for the UCI Appliances Energy dataset."""

    timestamp_col: str = "date"
    target_col: str = "Appliances"
    auxiliary_load_col: str = "lights"

    # T6 and RH_6 are outside the building (North side); T1-T5, T7-T9 are indoor rooms
    indoor_temp_cols: Tuple[str, ...] = (
        "T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"
    )
    indoor_humidity_cols: Tuple[str, ...] = (
        "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"
    )
    envelope_sensor_cols: Tuple[str, ...] = ("T6", "RH_6")
    weather_station_cols: Tuple[str, ...] = (
        "T_out",
        "Press_mm_hg",
        "RH_out",
        "Windspeed",
        "Visibility",
        "Tdewpoint",
    )
    noise_cols: Tuple[str, ...] = ("rv1", "rv2")

    @property
    def required_columns(self) -> List[str]:
        """Returns all expected columns in the raw CSV."""
        return [
            self.timestamp_col,
            self.target_col,
            self.auxiliary_load_col,
            *self.indoor_temp_cols,
            *self.indoor_humidity_cols,
            *self.envelope_sensor_cols,
            *self.weather_station_cols,
            *self.noise_cols,
        ]


@dataclass
class DataCleanerConfig:
    """Configuration parameters for data cleaning and preparation."""

    expected_freq: str = "10min"
    max_ffill_steps: int = 6  # Max 1 hour of forward-filling for sensor dropouts
    drop_noise_cols: bool = True
    drop_raw_lights: bool = False  # Keep if lagging in feature step; drop raw before fit
    clip_physical_bounds: bool = True


class EnergyDataProcessor:
    """
    Cleans and prepares raw UCI appliance energy and environmental sensor data.
    Guarantees chronological ordering and strictly causal missing-value handling.
    """

    def __init__(
        self,
        schema: Optional[UCIDatasetSchema] = None,
        config: Optional[DataCleanerConfig] = None,
    ) -> None:
        self.schema = schema or UCIDatasetSchema()
        self.config = config or DataCleanerConfig()

    def load_and_prepare(self, file_path: str | Path) -> pd.DataFrame:
        """
        Executes the full data quality and sensor preparation workflow.

        Args:
            file_path: Path to the raw `energydata_complete.csv` file.

        Returns:
            A chronologically indexed, cleaned pandas DataFrame at 10-minute intervals.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

        logger.info("Loading raw energy dataset from %s", path)
        df = pd.read_csv(path)

        self._validate_schema(df)
        df = self._parse_and_align_timestamps(df)
        df = self._drop_unwanted_columns(df)
        df = self._coerce_numeric_types(df)
        df = self._enforce_physical_bounds(df)
        df = self._impute_missing_causally(df)

        logger.info(
            "Data preparation complete. Final shape: %s | Time span: %s to %s",
            df.shape,
            df.index.min(),
            df.index.max(),
        )
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Verifies that all required UCI columns are present."""
        missing_cols = set(self.schema.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Dataset is missing expected UCI schema columns: {sorted(missing_cols)}"
            )

    def _parse_and_align_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses timestamps, removes duplicate timestamps, sorts chronologically,
        and aligns data to a strict 10-minute frequency grid.
        """
        df = df.copy()
        ts_col = self.schema.timestamp_col

        df[ts_col] = pd.to_datetime(df[ts_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        invalid_ts_count = int(df[ts_col].isna().sum())
        if invalid_ts_count:
            logger.warning("Dropping %d rows with unparseable timestamps.", invalid_ts_count)
            df = df.dropna(subset=[ts_col])

        # Sort chronologically and deduplicate
        df = df.sort_values(by=ts_col, ascending=True)
        dup_count = int(df.duplicated(subset=[ts_col]).sum())
        if dup_count:
            logger.warning("Removing %d duplicate timestamps (keeping first).", dup_count)
            df = df.drop_duplicates(subset=[ts_col], keep="first")

        df = df.set_index(ts_col)

        # Snap to uniform 10-minute grid to expose any missing time intervals
        full_grid = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=self.config.expected_freq,
            name=ts_col,
        )
        inserted_gaps = len(full_grid) - len(df)
        if inserted_gaps:
            logger.info(
                "Aligned to %s grid; exposed %d missing timestamp slots for causal imputation.",
                self.config.expected_freq,
                inserted_gaps,
            )
        df = df.reindex(full_grid)
        return df

    def _drop_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes synthetic noise variables (rv1, rv2) and optionally raw lights."""
        cols_to_drop: List[str] = []
        if self.config.drop_noise_cols:
            cols_to_drop.extend(
                [c for c in self.schema.noise_cols if c in df.columns]
            )
        if self.config.drop_raw_lights and self.schema.auxiliary_load_col in df.columns:
            cols_to_drop.append(self.schema.auxiliary_load_col)

        if cols_to_drop:
            logger.info("Dropping columns: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)
        return df

    def _coerce_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all sensor and target columns are float64."""
        df = df.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        return df

    def _enforce_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clips impossible physical sensor readings (e.g., negative energy Wh or
        relative humidity outside [0%, 100%]).
        """
        if not self.config.clip_physical_bounds:
            return df

        df = df.copy()
        # Energy consumption cannot be negative
        if self.schema.target_col in df.columns:
            df[self.schema.target_col] = df[self.schema.target_col].clip(lower=0.0)
        if self.schema.auxiliary_load_col in df.columns:
            df[self.schema.auxiliary_load_col] = df[self.schema.auxiliary_load_col].clip(lower=0.0)

        # Relative humidity sensors must lie within [0.0, 100.0]
        humidity_cols = [
            *self.schema.indoor_humidity_cols,
            "RH_6",
            "RH_out",
        ]
        for col in humidity_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.0, upper=100.0)

        # Windspeed and Visibility cannot be negative
        for col in ("Windspeed", "Visibility"):
            if col in df.columns:
                df[col] = df[col].clip(lower=0.0)

        return df

    def _impute_missing_causally(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values strictly causally (using only past observations).
        Never uses backward-fill (bfill) or global statistics to prevent look-ahead leakage.
        """
        df = df.copy()
        total_missing_before = int(df.isna().sum().sum())
        if total_missing_before == 0:
            return df

        logger.info("Causally imputing %d missing cell values.", total_missing_before)

        # 1. Forward-fill short sensor dropouts up to max_ffill_steps (e.g., 60 mins)
        df = df.ffill(limit=self.config.max_ffill_steps)

        # 2. For longer gaps, fill with expanding median (computed strictly up to time t)
        remaining_na = int(df.isna().sum().sum())
        if remaining_na:
            logger.warning(
                "%d values remain missing after forward-fill; applying causal expanding median.",
                remaining_na,
            )
            expanding_medians = df.expanding(min_periods=1).median()
            df = df.fillna(expanding_medians)

        # 3. Drop any leading rows if the very first timestamp had NaNs
        df = df.dropna(how="any")
        return df