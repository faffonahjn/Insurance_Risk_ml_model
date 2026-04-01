"""
Data loader — reads raw CSV, validates schema, returns clean DataFrame.
"""
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "age", "sex", "bmi", "children", "smoker", "region",
    "is_high_risk", "bmi_age_interaction", "sex_female",
    "smoker_flag", "region_northeast", "region_northwest",
    "region_southeast", "region_southwest", "age_group", "bmi_category",
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} records from {path}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def split_features_target(df: pd.DataFrame, config: dict):
    target = config["data"]["target"]
    drop_cols = config["data"]["drop_columns"]

    # Drop leakage columns (only those present in df)
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]

    logger.info(f"Features: {X.shape[1]} | Target: {target} | Class balance: {y.mean():.2%} positive")
    return X, y
