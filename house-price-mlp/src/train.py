"""Train/tune the required sklearn MLPRegressor house-price model."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from .data_loader import ensure_csv, ensure_images, load_dataset
from .image_features import extract_image_feature_frame
from .preprocessing import TARGET_COLUMN, build_preprocessor, select_features
from .evaluate import create_evaluation_artifacts, metric_dict
from .prediction_utils import clip_predictions

RANDOM_STATE = 42


def make_estimator(features: pd.DataFrame, image_columns: list[str]) -> TransformedTargetRegressor:
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(features, image_columns)),
        ("mlp", MLPRegressor(
            activation="relu", solver="adam", learning_rate_init=0.001,
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=RANDOM_STATE, batch_size=128,
            tol=5e-4,
        )),
    ])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1,
        check_inverse=False,
    )


def _get_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    actual = np.asarray(actual, dtype=float)
    return {
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "mse_usd_squared": float(mean_squared_error(actual, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train_model(data_dir: Path, model_path: Path, outputs_dir: Path, use_images: bool = False,
                n_iter: int = 4, cv: int = 3) -> dict:
    """Train a tuned MLP and save the estimator, metadata, metrics, and plots."""
    data_dir, model_path, outputs_dir = Path(data_dir), Path(model_path), Path(outputs_dir)
    csv_path = ensure_csv(data_dir)
    frame = load_dataset(csv_path)
    image_columns: list[str] = []
    if use_images:
        image_dir = ensure_images(data_dir)
        image_frame = extract_image_feature_frame(frame, image_dir)
        image_columns = image_frame.columns.tolist()
        X = pd.concat([select_features(frame), image_frame], axis=1)
    else:
        X = select_features(frame)
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype(float)

    # Fixed, independent partitions: 65% train, 15% validation, 20% test.
    row_ids = np.arange(len(frame))
    trainval_ids, test_ids = train_test_split(row_ids, test_size=0.20, random_state=RANDOM_STATE)
    train_ids, validation_ids = train_test_split(
        trainval_ids, test_size=0.1875, random_state=RANDOM_STATE,
    )
    X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
    X_val, y_val = X.iloc[validation_ids], y.iloc[validation_ids]
    X_test, y_test = X.iloc[test_ids], y.iloc[test_ids]
    target_bounds = (float(y_train.min()), float(y_train.max()))

    estimator = make_estimator(X_train, image_columns)
    parameter_space = {
        "regressor__mlp__hidden_layer_sizes": [(64, 32), (100, 50), (128, 64)],
        "regressor__mlp__learning_rate_init": [0.0005, 0.001],
        "regressor__mlp__alpha": [0.0001, 0.001, 0.01],
        "regressor__mlp__batch_size": [128, 256],
    }
    search = RandomizedSearchCV(
        estimator=estimator, param_distributions=parameter_space,
        n_iter=max(1, min(int(n_iter), 8)), scoring="neg_mean_absolute_error",
        cv=max(2, int(cv)), random_state=RANDOM_STATE, n_jobs=1,
        refit=True, verbose=1, error_score="raise",
    )
    print(f"Training on {len(train_ids):,} rows; tuning MLPRegressor with {search.n_iter} candidates x {search.cv} folds.")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    # An unconstrained MLP can extrapolate far outside its observed price domain
    # for unusual inputs (e.g. an extreme bathroom count). Keep outputs realistic.
    val_prediction = clip_predictions(best_model.predict(X_val), target_bounds)
    test_prediction = clip_predictions(best_model.predict(X_test), target_bounds)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "project": "House Price Prediction using Multilayer Perceptron (MLP)",
        "dataset_csv": str(csv_path.name),
        "target_column": TARGET_COLUMN,
        "target_unit": "USD",
        "target_transform": "log1p(price) during fit; predictions inverse-transformed with expm1",
        "prediction_bounds_usd": {"minimum": target_bounds[0], "maximum": target_bounds[1]},
        "prediction_postprocessing": "Clip inverse-transformed predictions to the minimum and maximum prices observed in the training split.",
        "rows_after_target_validation": int(len(frame)),
        "split_rows": {"train": int(len(train_ids)), "validation": int(len(validation_ids)), "test": int(len(test_ids))},
        "missing_values_by_column": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "price_skewness_raw": float(y.skew()),
        "price_skewness_log1p": float(np.log1p(y).skew()),
        "features_used": list(X.columns),
        "image_features_used": bool(use_images),
        "image_feature_count": int(len(image_columns)),
        "random_state": RANDOM_STATE,
        "model": "sklearn.neural_network.MLPRegressor",
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
        "best_cv_mae_usd": float(-search.best_score_),
        "validation_metrics": _get_metrics(y_val, val_prediction),
        "test_metrics": _get_metrics(y_test, test_prediction),
    }
    bundle = {
        "estimator": best_model,
        "target_column": TARGET_COLUMN,
        "features": list(X.columns),
        "image_features": image_columns,
        "uses_images": bool(use_images),
        "target_bounds_usd": target_bounds,
        "image_directory": "socal_pics",
        "metadata": {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_transform": "log1p/expm1",
        "target_bounds_usd": target_bounds,
            "random_state": RANDOM_STATE,
            "selected_features": list(X.columns),
            "best_params": metrics["best_params"],
            "best_cv_mae_usd": float(-search.best_score_),
        },
    }
    joblib.dump(bundle, model_path)
    with (outputs_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    create_evaluation_artifacts(
        y_test.to_numpy(), test_prediction, y.to_numpy(), best_model, outputs_dir
    )
    print("\nTest metrics (original USD scale):")
    for name, value in metrics["test_metrics"].items():
        print(f"  {name}: {value:,.4f}")
    print(f"\nSaved model: {model_path}\nSaved outputs: {outputs_dir}")
    return metrics
