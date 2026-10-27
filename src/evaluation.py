"""
src/evaluation.py

Holdout evaluation, multi-model leaderboard generation, unified feature
importance extraction, and residual diagnostics for GridSense ML.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class EnergyModelEvaluator:
    """
    Evaluates trained forecasting models on chronological holdout data,
    extracts feature importances, and prepares diagnostic frames for Streamlit.
    """

    @staticmethod
    def compute_regression_metrics(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculates MAE, RMSE, R2, and Mean Bias Error (MBE) in Watt-hours (Wh).
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        r2 = float(r2_score(y_t, y_p))
        mbe = float(np.mean(y_p - y_t))

        return {
            "test_mae": round(mae, 4),
            "test_rmse": round(rmse, 4),
            "test_r2": round(r2, 4),
            "test_bias_wh": round(mbe, 4),
        }

    def evaluate_all_models(
        self,
        models: Dict[str, BaseEstimator],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_results: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs holdout predictions for all registered models, compiles a ranked
        benchmarking leaderboard, and returns a timestamp-aligned predictions frame.

        Args:
            models: Dictionary mapping model names to fitted estimators.
            X_test: Holdout feature matrix.
            y_test: Holdout target series (`Appliances` in Wh).
            cv_results: Optional DataFrame from `run_timeseries_backtest`.

        Returns:
            Tuple of:
              - leaderboard_df: Ranked summary table with CV and Holdout metrics.
              - predictions_df: Timestamp-indexed DataFrame of actuals, predictions, and residuals.
        """
        metric_rows: List[Dict[str, Any]] = []
        predictions_df = pd.DataFrame({"actual_wh": y_test}, index=X_test.index)

        for model_name, estimator in models.items():
            # Enforce non-negative physical energy forecasts
            raw_preds = estimator.predict(X_test)
            clipped_preds = np.clip(raw_preds, a_min=0.0, a_max=None)

            metrics = self.compute_regression_metrics(y_test, clipped_preds)
            metrics["model"] = model_name
            metric_rows.append(metrics)

            predictions_df[f"pred_{model_name}"] = clipped_preds
            predictions_df[f"residual_{model_name}"] = y_test - clipped_preds
            predictions_df[f"abs_error_{model_name}"] = np.abs(y_test - clipped_preds)

            logger.info(
                "[Holdout] %-24s | RMSE: %.2f Wh | MAE: %.2f Wh | R2: %.4f",
                model_name,
                metrics["test_rmse"],
                metrics["test_mae"],
                metrics["test_r2"],
            )

        leaderboard_df = pd.DataFrame(metric_rows)

        if cv_results is not None and not cv_results.empty:
            leaderboard_df = pd.merge(
                cv_results, leaderboard_df, on="model", how="right"
            )

        # Calculate percentage RMSE improvement over the Persistence Baseline
        baseline_rows = leaderboard_df.loc[
            leaderboard_df["model"].str.contains("Baseline", case=False, na=False),
            "test_rmse",
        ]
        if not baseline_rows.empty:
            baseline_rmse = float(baseline_rows.iloc[0])
            leaderboard_df["rmse_skill_vs_baseline_pct"] = np.round(
                (1.0 - (leaderboard_df["test_rmse"] / baseline_rmse)) * 100.0, 2
            )

        leaderboard_df = leaderboard_df.sort_values(
            by="test_rmse", ascending=True
        ).reset_index(drop=True)

        return leaderboard_df, predictions_df

    @staticmethod
    def extract_feature_importance(
        estimator: BaseEstimator,
        feature_names: Sequence[str],
        top_n: Optional[int] = 25,
    ) -> pd.DataFrame:
        """
        Extracts and normalizes feature importances across heterogeneous models:
          - Tree models (Random Forest, XGBoost): `feature_importances_`
          - Scaled Linear Pipelines: absolute standardized coefficients `|coef_|`

        Args:
            estimator: Fitted Scikit-Learn estimator or Pipeline.
            feature_names: Ordered list of input feature names.
            top_n: Number of top features to return (None returns all).

        Returns:
            DataFrame with columns `['feature', 'raw_importance', 'importance_pct']`.
        """
        target_model = estimator
        if isinstance(estimator, Pipeline):
            target_model = estimator.named_steps.get(
                "regressor", estimator.steps[-1][1]
            )

        if hasattr(target_model, "feature_importances_"):
            raw_scores = np.asarray(target_model.feature_importances_, dtype=np.float64)
        elif hasattr(target_model, "coef_"):
            raw_scores = np.abs(np.asarray(target_model.coef_, dtype=np.float64)).ravel()
        else:
            logger.warning(
                "Estimator %s does not expose feature_importances_ or coef_.",
                type(target_model).__name__,
            )
            return pd.DataFrame(columns=["feature", "raw_importance", "importance_pct"])

        if len(raw_scores) != len(feature_names):
            raise ValueError(
                f"Length mismatch: {len(raw_scores)} importance scores vs "
                f"{len(feature_names)} feature names."
            )

        total_score = float(np.sum(raw_scores))
        norm_pct = (
            (raw_scores / total_score) * 100.0
            if total_score > 0.0
            else np.zeros_like(raw_scores)
        )

        importance_df = pd.DataFrame(
            {
                "feature": list(feature_names),
                "raw_importance": raw_scores,
                "importance_pct": np.round(norm_pct, 4),
            }
        ).sort_values(by="raw_importance", ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df.reset_index(drop=True)

    def extract_all_importances(
        self,
        models: Dict[str, BaseEstimator],
        feature_names: Sequence[str],
        top_n: int = 25,
    ) -> Dict[str, pd.DataFrame]:
        """
        Extracts feature importance DataFrames for all models that support attribution.
        """
        importances: Dict[str, pd.DataFrame] = {}
        for model_name, estimator in models.items():
            df_imp = self.extract_feature_importance(
                estimator=estimator,
                feature_names=feature_names,
                top_n=top_n,
            )
            if not df_imp.empty:
                importances[model_name] = df_imp
        return importances

    @staticmethod
    def compute_hourly_error_profile(
        predictions_df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """
        Aggregates MAE and RMSE by hour of the day (0-23) to diagnose whether errors
        concentrate during morning ramps or evening peak-demand periods.
        """
        pred_col = f"pred_{model_name}"
        if pred_col not in predictions_df.columns:
            raise KeyError(f"Prediction column '{pred_col}' not found.")

        temp = predictions_df[["actual_wh", pred_col]].copy()
        temp["hour"] = temp.index.hour
        temp["abs_err"] = np.abs(temp["actual_wh"] - temp[pred_col])
        temp["sq_err"] = (temp["actual_wh"] - temp[pred_col]) ** 2

        hourly = (
            temp.groupby("hour", as_index=False)
            .agg(
                mean_actual_wh=("actual_wh", "mean"),
                mean_pred_wh=(pred_col, "mean"),
                mae_wh=("abs_err", "mean"),
                mse_wh=("sq_err", "mean"),
            )
        )
        hourly["rmse_wh"] = np.sqrt(hourly["mse_wh"])
        return hourly.drop(columns=["mse_wh"])

    @staticmethod
    def save_evaluation_artifacts(
        leaderboard_df: pd.DataFrame,
        importances_map: Dict[str, pd.DataFrame],
        predictions_df: pd.DataFrame,
        output_dir: str | Path = "artifacts/metrics",
    ) -> None:
        """Saves evaluation tables and predictions for fast loading in Streamlit."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        leaderboard_df.to_csv(out_path / "model_leaderboard.csv", index=False)
        predictions_df.to_parquet(out_path / "holdout_predictions.parquet")

        importances_serializable = {
            model_name: df.to_dict(orient="records")
            for model_name, df in importances_map.items()
        }
        with open(out_path / "feature_importances.json", "w", encoding="utf-8") as f:
            json.dump(importances_serializable, f, indent=2)

        logger.info("Saved evaluation leaderboard, predictions, and importances to %s", out_path)