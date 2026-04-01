"""Unit tests — model trainer."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_config, load_raw_data, split_features_target
from src.models.trainer import build_pipeline, train
from sklearn.model_selection import train_test_split


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


@pytest.fixture
def train_test(config):
    df = load_raw_data(config["paths"]["raw_data"])
    X, y = split_features_target(df, config)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def test_pipeline_builds(config):
    pipeline = build_pipeline(config)
    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_pipeline_fits_and_predicts(config, train_test):
    X_train, X_test, y_train, y_test = train_test
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    assert len(preds) == len(X_test)
    assert probs.shape == (len(X_test), 2)
    assert set(preds).issubset({0, 1})


def test_train_returns_valid_metrics(config):
    df = load_raw_data(config["paths"]["raw_data"])
    X, y = split_features_target(df, config)
    _, metrics, X_test, y_test = train(X, y, config)
    assert "test_roc_auc" in metrics
    assert "cv_auc_mean" in metrics
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
