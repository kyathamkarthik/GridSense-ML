"""
src/peak_intelligence.py

Calculates leakage-free peak demand thresholds from historical training data
and computes binary classification metrics (Precision, Recall, F1) to evaluate
each model's ability to forecast high-strain grid intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class PeakThresholdConfig:
    """Configuration for peak demand classification."""
    quantile_threshold: float = 0.90  # Default: Top 10% of training demand is a 'Peak'


class PeakDemandClassifier:
    """
    Identifies high-usage intervals and evaluates model classification performance
    for demand-response planning.
    """

    def __init__(self, config: PeakThresholdConfig | None = None) -> None:
        self.config = config or PeakThresholdConfig()
        self.peak_threshold_wh_: float = 0.0

    def fit_threshold(self, y_train: pd.Series) -> float:
        """
        Derives the peak threshold strictly from the training partition to prevent
        look-ahead leakage.
        """
        if y_train.empty:
            raise ValueError("Training series is empty; cannot compute peak threshold.")
        
        self.peak_threshold_wh_ = float(y_train.quantile(self.config.quantile_threshold))
        logger.info(
            "Fitted peak threshold at the %dth percentile: %.2f Wh",
            int(self.config.quantile_threshold * 100),
            self.peak_threshold_wh_,
        )
        return self.peak_threshold_wh_

    def evaluate_peak_forecasts(
        self,
        predictions_df: pd.DataFrame,
        actual_col: str = "actual_wh",
    ) -> pd.DataFrame:
        """
        Calculates precision, recall, and F1 scores for every model in the predictions
        DataFrame based on the fitted training threshold.

        Args:
            predictions_df: Timestamp-indexed DataFrame containing actuals and predictions.
            actual_col: Name of the ground-truth column in `predictions_df`.

        Returns:
            DataFrame containing binary classification metrics for each model.
        """
        if self.peak_threshold_wh_ <= 0.0:
            raise RuntimeError("Peak threshold not fitted. Call `fit_threshold` first.")

        if actual_col not in predictions_df.columns:
            raise KeyError(f"Actuals column '{actual_col}' not found in predictions DataFrame.")

        # Binarize ground truth
        y_true_binary = (predictions_df[actual_col] >= self.peak_threshold_wh_).astype(int)
        peak_count = y_true_binary.sum()
        logger.info(
            "Evaluating peak intelligence against %d true peak events in the holdout set.", 
            peak_count
        )

        pred_cols = [c for c in predictions_df.columns if c.startswith("pred_")]
        metrics_list: List[Dict[str, Any]] = []

        for col in pred_cols:
            model_name = col.replace("pred_", "")
            
            # Binarize model forecasts
            y_pred_binary = (predictions_df[col] >= self.peak_threshold_wh_).astype(int)
            
            precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
            recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
            f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
            
            # Extract true positives, false positives, false negatives
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

            metrics_list.append({
                "model": model_name,
                "peak_precision": round(precision, 4),
                "peak_recall": round(recall, 4),
                "peak_f1": round(f1, 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            })
            
            logger.info(
                "[Peak] %-24s | Precision: %.2f | Recall: %.2f | F1: %.2f",
                model_name,
                precision,
                recall,
                f1,
            )

        return pd.DataFrame(metrics_list).sort_values(by="peak_f1", ascending=False).reset_index(drop=True)