"""
train_pipeline.py

End-to-end orchestration CLI for the GridSense ML forecasting pipeline.
Executes data ingestion, feature engineering, backtesting, peak analytics,
and artifact serialization.
"""

import argparse
import json
import logging
from pathlib import Path
import warnings

from src.data_processing import EnergyDataProcessor
from src.features import EnergyFeatureEngineer
from src.models import EnergyModelTrainer
from src.evaluation import EnergyModelEvaluator
from src.peak_intelligence import PeakDemandClassifier, PeakThresholdConfig

# Suppress overly verbose Scikit-Learn/XGBoost warnings for clean CLI output
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridSense ML End-to-End Training Pipeline")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/raw/energydata_complete.csv",
        help="Path to the raw UCI dataset."
    )
    parser.add_argument(
        "--artifact_dir", 
        type=str, 
        default="artifacts",
        help="Root directory to save models and metrics."
    )
    parser.add_argument(
        "--peak_quantile", 
        type=float, 
        default=0.90,
        help="Training distribution quantile to classify as peak demand."
    )
    parser.add_argument(
        "--skip_cv", 
        action="store_true",
        help="Skip the TimeSeriesSplit cross-validation to save time."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== GridSense ML Pipeline Initiated ===")

    # 1. Data Preparation
    logger.info("--- Step 1: Data Preparation ---")
    processor = EnergyDataProcessor()
    clean_df = processor.load_and_prepare(args.data_path)

    # 2. Feature Engineering
    logger.info("--- Step 2: Causal Feature Engineering ---")
    engineer = EnergyFeatureEngineer()
    feature_df = engineer.fit_transform(clean_df)

    # 3. Model Training & Splitting
    logger.info("--- Step 3: Training & Partitioning ---")
    trainer = EnergyModelTrainer()
    split = trainer.split_chronologically(feature_df)
    
    cv_df = None
    if not args.skip_cv:
        cv_df = trainer.run_timeseries_backtest(split.X_train, split.y_train)
    else:
        logger.info("Skipping TimeSeriesSplit backtesting (--skip_cv flag passed).")

    models = trainer.fit_models(split.X_train, split.y_train)
    trainer.save_models(output_dir=artifact_path / "models")

    # 4. Evaluation & Diagnostics
    logger.info("--- Step 4: Holdout Evaluation ---")
    evaluator = EnergyModelEvaluator()
    leaderboard_df, predictions_df = evaluator.evaluate_all_models(
        models=models,
        X_test=split.X_test,
        y_test=split.y_test,
        cv_results=cv_df,
    )
    
    importances_map = evaluator.extract_all_importances(
        models, 
        engineer.feature_names_out_
    )
    evaluator.save_evaluation_artifacts(
        leaderboard_df=leaderboard_df,
        importances_map=importances_map,
        predictions_df=predictions_df,
        output_dir=artifact_path / "metrics"
    )

    # 5. Peak Intelligence Analytics
    logger.info("--- Step 5: Peak Intelligence Analytics ---")
    peak_classifier = PeakDemandClassifier(
        config=PeakThresholdConfig(quantile_threshold=args.peak_quantile)
    )
    
    # Fit strictly on train to prevent leakage
    threshold_wh = peak_classifier.fit_threshold(split.y_train)
    peak_metrics_df = peak_classifier.evaluate_peak_forecasts(predictions_df)
    
    # Merge peak metrics into the main leaderboard and save
    final_leaderboard = pd.merge(leaderboard_df, peak_metrics_df, on="model", how="left")
    final_leaderboard.to_csv(artifact_path / "metrics" / "model_leaderboard.csv", index=False)
    
    # Save the physical threshold value for the Streamlit UI to use
    with open(artifact_path / "metrics" / "peak_threshold.json", "w") as f:
        json.dump({"threshold_wh": threshold_wh, "quantile": args.peak_quantile}, f)

    logger.info("=== Pipeline Execution Complete ===")
    logger.info("\nFinal Leaderboard (Top 3 by RMSE):\n%s", 
                final_leaderboard[["model", "test_rmse", "peak_f1"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()