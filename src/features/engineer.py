"""
Feature engineering — encodes categoricals, scales numerics, builds final feature matrix.
All transformers fit on train only to prevent leakage.
"""
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessor(config: dict) -> ColumnTransformer:
    cat_features = config["data"]["categorical_features"]
    num_features = config["data"]["numeric_features"]
    bin_features = config["data"].get("binary_features", [])

    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    transformers = [
        ("cat", categorical_pipe, cat_features),
        ("num", numeric_pipe, num_features),
    ]
    if bin_features:
        transformers.append(("bin", "passthrough", bin_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(f"Preprocessor | cat={len(cat_features)} | num={len(num_features)} | bin={len(bin_features)}")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    return list(preprocessor.get_feature_names_out())
