"""
predict_pipeline.py — batch inference on a CSV file.

Usage:
    python pipelines/predict_pipeline.py --input data/raw/medical_insurance.csv --output data/processed/predictions.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

DECISION_THRESHOLD = 0.35

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_config
from src.models.trainer import load_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(input_path: str, output_path: str, config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    model_path = Path(config["paths"]["model_dir"]) / config["training"]["model_filename"]

    logger.info(f"Loading model from {model_path}")
    pipeline = load_pipeline(str(model_path))

    logger.info(f"Loading input data from {input_path}")
    df = pd.read_csv(input_path)

    # Drop leakage columns if present
    drop_cols = config["data"]["drop_columns"] + [config["data"]["target"]]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)

    df["predicted_risk_prob"] = probs.round(4)
    df["predicted_is_high_risk"] = preds
    df["risk_label"] = df["predicted_is_high_risk"].map({1: "High Risk", 0: "Low Risk"})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved → {output_path} | High risk: {preds.sum()}/{len(preds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/predictions.csv")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.input, args.output, args.config)
