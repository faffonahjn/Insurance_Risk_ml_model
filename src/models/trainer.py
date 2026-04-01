"""
Model training — XGBoost classifier with cross-validation, threshold tuning,
and joblib persistence.
"""
import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.engineer import build_preprocessor

logger = logging.getLogger(__name__)


def build_pipeline(config: dict) -> Pipeline:
    preprocessor = build_preprocessor(config)
    model_params = config["model"]["params"].copy()
    model_params.pop("use_label_encoder", None)

    clf = XGBClassifier(**model_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    return pipeline


def train(X: pd.DataFrame, y: pd.Series, config: dict) -> Tuple[Pipeline, dict]:
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(config)

    # Cross-validation on train set
    cv = StratifiedKFold(n_splits=config["model"]["cv_folds"], shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=config["model"]["scoring"])
    logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on full train set
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "test_avg_precision": round(float(average_precision_score(y_test, y_prob)), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    logger.info(f"Test AUC: {metrics['test_roc_auc']} | AP: {metrics['test_avg_precision']}")
    return pipeline, metrics, X_test, y_test


def save_pipeline(pipeline: Pipeline, config: dict) -> Path:
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config["training"]["model_filename"]
   
   
    joblib.dump(pipeline, model_path)
    
    # Save XGBoost booster natively to eliminate version warnings
    booster_path = model_dir / "xgb_booster.json"
    pipeline.named_steps["classifier"].get_booster().save_model(str(booster_path))
    
    logger.info(f"Model saved -> {model_path}")
    return model_path


def load_pipeline(model_path: str) -> Pipeline:
    return joblib.load(model_path)


def save_metrics(metrics: dict, config: dict):
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "evaluation_metrics.json"
    # Remove non-serializable parts for top-level JSON
    export = {k: v for k, v in metrics.items() if k != "classification_report"}
    export["classification_report"] = metrics["classification_report"]
    with open(metrics_path, "w") as f:
        json.dump(export, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")
