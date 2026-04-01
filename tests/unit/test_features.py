"""Unit tests — feature engineering pipeline."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_config, load_raw_data, split_features_target
from src.features.engineer import build_preprocessor


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


@pytest.fixture
def sample_data(config):
    df = load_raw_data(config["paths"]["raw_data"])
    return df.head(50)


def test_preprocessor_builds(config):
    preprocessor = build_preprocessor(config)
    assert preprocessor is not None


def test_feature_target_split(config, sample_data):
    X, y = split_features_target(sample_data, config)
    assert "is_high_risk" not in X.columns
    assert y.name == "is_high_risk"
    assert len(X) == len(y)


def test_no_leakage_columns(config, sample_data):
    X, y = split_features_target(sample_data, config)
    leakage_cols = config["data"]["drop_columns"]
    for col in leakage_cols:
        assert col not in X.columns, f"Leakage column found: {col}"


def test_preprocessor_transform_shape(config, sample_data):
    X, y = split_features_target(sample_data, config)
    preprocessor = build_preprocessor(config)
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > 0
