"""
train_pipeline.py — orchestrates end-to-end training.

Usage:
    python pipelines/train_pipeline.py
    python pipelines/train_pipeline.py --config configs/config.yaml
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import logging
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_config, load_raw_data, split_features_target
from src.models.trainer import train, save_pipeline, save_metrics
from src.evaluation.metrics import run_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log"),
    ],
)
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/config.yaml"):
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE START")
    logger.info("=" * 60)

    # 1. Load config
    config = load_config(config_path)

    # 2. Load data
    df = load_raw_data(config["paths"]["raw_data"])

    # 3. Split features / target
    X, y = split_features_target(df, config)

    # 4. Train + evaluate
    pipeline, metrics, X_test, y_test = train(X, y, config)

    # 5. Save model
    save_pipeline(pipeline, config)

    # 6. Save metrics
    save_metrics(metrics, config)

    # 7. Save evaluation plots
    run_all_plots(pipeline, X_test, y_test, config)

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE | AUC: {metrics['test_roc_auc']} | AP: {metrics['test_avg_precision']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
