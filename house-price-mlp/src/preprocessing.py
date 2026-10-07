"""Leakage-safe preprocessing pipelines for the house-price MLP."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "price"
IDENTIFIER_COLUMNS = {"price", "image_id", "street", "n_citi"}


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs based on the inspected dataset, excluding IDs/leaky proxies."""
    excluded = {name.lower() for name in IDENTIFIER_COLUMNS}
    selected = [column for column in frame.columns if column.lower() not in excluded]
    if not selected:
        raise ValueError("No usable feature columns remain after excluding IDs and target.")
    return frame[selected].copy()


def build_preprocessor(features: pd.DataFrame, image_feature_columns: Iterable[str] = ()) -> ColumnTransformer:
    """Create imputing/scaling/encoding transformers; fit occurs only within training folds."""
    image_cols = set(image_feature_columns)
    hog_cols = [c for c in features.columns if c in image_cols and c.startswith("hog_")]
    image_stats = [c for c in features.columns if c in image_cols and c not in hog_cols]
    other_cols = [c for c in features.columns if c not in image_cols]
    numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(features[c])]
    categorical_cols = [c for c in other_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]), categorical_cols))
    if image_stats:
        transformers.append(("image_stats", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), image_stats))
    if hog_cols:
        # PCA is learned inside each training fold; image descriptors never see validation/test rows.
        transformers.append(("image_hog", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=24, random_state=42, svd_solver="randomized")),
        ]), hog_cols))
    if not transformers:
        raise ValueError("No features were available to preprocess.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
