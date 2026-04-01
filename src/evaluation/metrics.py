"""
Evaluation utilities — plots ROC, PR curve, feature importance, confusion matrix.
All artifacts saved to artifacts/plots/.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _plot_dir(config: dict) -> Path:
    d = Path(config["paths"]["plots_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_roc_curve(pipeline: Pipeline, X_test, y_test, config: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve — Insurance Risk Classifier")
    path = _plot_dir(config) / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved → {path}")


def plot_pr_curve(pipeline: Pipeline, X_test, y_test, config: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve — Insurance Risk Classifier")
    path = _plot_dir(config) / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved → {path}")


def plot_confusion_matrix(pipeline: Pipeline, X_test, y_test, config: dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=["Low Risk", "High Risk"],
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    path = _plot_dir(config) / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {path}")


def plot_feature_importance(pipeline: Pipeline, config: dict, top_n: int = 20):
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = list(preprocessor.get_feature_names_out())
    importances = clf.feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#1f77b4")
    ax.set_xlabel("XGBoost Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Features — Insurance Risk Classifier")
    path = _plot_dir(config) / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance saved → {path}")


def run_all_plots(pipeline: Pipeline, X_test, y_test, config: dict):
    plot_roc_curve(pipeline, X_test, y_test, config)
    plot_pr_curve(pipeline, X_test, y_test, config)
    plot_confusion_matrix(pipeline, X_test, y_test, config)
    plot_feature_importance(pipeline, config)
