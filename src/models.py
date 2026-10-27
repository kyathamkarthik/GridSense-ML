"""
src/models.py

Chronological dataset splitting, expanding-window TimeSeriesSplit backtesting,
and multi-model training registry (Baseline, Linear Regression, Random Forest,
and XGBoost) for the GridSense ML pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Container for leak-free chronological train and test partitions."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_span(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.X_train.index.min(), self.X_train.index.max()

    @property
    def test_span(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.X_test.index.min(), self.X_test.index.max()


@dataclass
class ModelSuiteConfig:
    """Configuration for chronological splitting, CV backtesting, and model hyperparameters."""

    target_col: str = "Appliances"
    baseline_feature_col: str = "Appliances_lag_1"
    test_size_ratio: float = 0.20

    # TimeSeriesSplit parameters (gap=6 enforces a 1-hour buffer between train and val folds)
    cv_splits: int = 5
    cv_gap: int = 6

    random_state: int = 42
    n_jobs: int = -1

    # Random Forest Hyperparameters
    rf_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 250,
            "max_depth": 16,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
        }
    )

    # XGBoost Hyperparameters
    xgb_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 3,
            "reg_alpha": 0.10,
            "reg_lambda": 1.00,
            "objective": "reg:squarederror",
        }
    )


class PersistenceBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Naive persistence baseline estimator compatible with Scikit-Learn pipelines.
    Predicts y_t using the most recent available lagged target (e.g., Appliances_lag_1).
    """

    def __init__(self, lag_col: str = "Appliances_lag_1") -> None:
        self.lag_col = lag_col
        self.fallback_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceBaselineRegressor":
        """Stores fallback training mean and verifies the persistence column exists."""
        self.fallback_mean_ = float(np.mean(y))
        if isinstance(X, pd.DataFrame) and self.lag_col not in X.columns:
            raise KeyError(
                f"Baseline lag column '{self.lag_col}' not found in feature matrix."
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the lagged target series as the persistence forecast."""
        if isinstance(X, pd.DataFrame) and self.lag_col in X.columns:
            preds = X[self.lag_col].to_numpy(dtype=np.float64)
            return np.nan_to_num(preds, nan=self.fallback_mean_)
        return np.full(shape=(len(X),), fill_value=self.fallback_mean_, dtype=np.float64)


class EnergyModelTrainer:
    """
    Orchestrates chronological data partitioning, expanding-window TimeSeriesSplit
    backtesting, final model fitting, and artifact persistence.
    """

    def __init__(self, config: Optional[ModelSuiteConfig] = None) -> None:
        self.config = config or ModelSuiteConfig()
        self.models: Dict[str, BaseEstimator] = {}
        self.cv_results_: pd.DataFrame = pd.DataFrame()

    def split_chronologically(self, df: pd.DataFrame) -> ChronologicalSplit:
        """
        Splits an engineered DataFrame into contiguous past (train) and future (test)
        sets without shuffling.

        Args:
            df: Engineered DataFrame indexed by a sorted DatetimeIndex.

        Returns:
            ChronologicalSplit instance containing X_train, X_test, y_train, y_test.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Feature DataFrame must be indexed by a DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index(ascending=True)

        target = self.config.target_col
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' is missing from the DataFrame.")

        n_total = len(df)
        split_idx = int(n_total * (1.0 - self.config.test_size_ratio))
        if split_idx <= 0 or split_idx >= n_total:
            raise ValueError(f"Invalid split_idx={split_idx} for dataset of length {n_total}.")

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target].copy()
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target].copy()

        split = ChronologicalSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        logger.info(
            "Chronological split complete | Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
            len(X_train),
            split.train_span[0],
            split.train_span[1],
            len(X_test),
            split.test_span[0],
            split.test_span[1],
        )
        return split

    def build_model_registry(self) -> Dict[str, BaseEstimator]:
        """
        Instantiates the four benchmark models:
          1. Persistence Baseline
          2. Scaled Linear Regression
          3. Random Forest Regressor
          4. XGBoost Regressor
        """
        registry: Dict[str, BaseEstimator] = {
            "Baseline (Persistence)": PersistenceBaselineRegressor(
                lag_col=self.config.baseline_feature_col
            ),
            "Linear Regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("regressor", LinearRegression()),
                ]
            ),
            "Random Forest": RandomForestRegressor(
                **self.config.rf_params,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            ),
            "XGBoost": XGBRegressor(
                **self.config.xgb_params,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            ),
        }
        return registry

    def run_timeseries_backtest(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """
        Performs expanding-window cross-validation using TimeSeriesSplit on the
        training partition to evaluate out-of-sample stability across temporal regimes.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.

        Returns:
            DataFrame summarizing fold-by-fold and aggregated CV metrics per model.
        """
        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_splits,
            gap=self.config.cv_gap,
        )
        registry = self.build_model_registry()
        records: List[Dict[str, Any]] = []

        logger.info(
            "Starting TimeSeriesSplit backtesting (%d splits, gap=%d steps)...",
            self.config.cv_splits,
            self.config.cv_gap,
        )

        for model_name, estimator in registry.items():
            fold_maes: List[float] = []
            fold_rmses: List[float] = []
            fold_r2s: List[float] = []

            for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                estimator.fit(X_tr, y_tr)
                val_preds = np.clip(estimator.predict(X_val), a_min=0.0, a_max=None)

                fold_maes.append(float(mean_absolute_error(y_val, val_preds)))
                fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, val_preds))))
                fold_r2s.append(float(r2_score(y_val, val_preds)))

            record = {
                "model": model_name,
                "cv_mae_mean": float(np.mean(fold_maes)),
                "cv_mae_std": float(np.std(fold_maes)),
                "cv_rmse_mean": float(np.mean(fold_rmses)),
                "cv_rmse_std": float(np.std(fold_rmses)),
                "cv_r2_mean": float(np.mean(fold_r2s)),
                "cv_r2_std": float(np.std(fold_r2s)),
            }
            records.append(record)
            logger.info(
                "[CV] %-24s | RMSE: %.2f (±%.2f) | MAE: %.2f | R2: %.3f",
                model_name,
                record["cv_rmse_mean"],
                record["cv_rmse_std"],
                record["cv_mae_mean"],
                record["cv_r2_mean"],
            )

        self.cv_results_ = pd.DataFrame(records)
        return self.cv_results_

    def fit_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, BaseEstimator]:
        """
        Fits all four models on the complete historical training partition.
        """
        self.models = self.build_model_registry()
        for model_name, estimator in self.models.items():
            logger.info("Fitting full training set for model: %s", model_name)
            estimator.fit(X_train, y_train)
        return self.models

    def save_models(self, output_dir: str | Path = "artifacts/models") -> Dict[str, Path]:
        """Serializes trained model objects to disk via joblib."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_paths: Dict[str, Path] = {}
        for model_name, estimator in self.models.items():
            slug = (
                model_name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            file_path = out_path / f"{slug}.joblib"
            joblib.dump(estimator, file_path)
            saved_paths[model_name] = file_path
            logger.info("Saved %s artifact to %s", model_name, file_path)

        return saved_paths