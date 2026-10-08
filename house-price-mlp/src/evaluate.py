"""Evaluation metrics and visual reports, all in the original USD target scale."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid", context="notebook")
from .prediction_utils import clip_predictions


def metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    mse = mean_squared_error(actual, predicted)
    return {
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "mse_usd_squared": float(mse),
        "rmse_usd": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def create_evaluation_artifacts(actual: np.ndarray, predicted: np.ndarray,
                                all_prices: np.ndarray, fitted_model, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    residuals = actual - predicted

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, predicted, alpha=0.55, s=22, color="#2563eb", edgecolors="none")
    bound = max(float(actual.max()), float(predicted.max()))
    ax.plot([0, bound], [0, bound], "--", color="#dc2626", linewidth=1.5, label="Perfect prediction")
    ax.set(title="Actual vs. Predicted House Prices", xlabel="Actual price (USD)", ylabel="Predicted price (USD)")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "actual_vs_predicted.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predicted, residuals, alpha=0.55, s=22, color="#0f766e", edgecolors="none")
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set(title="Residuals vs. Predicted Price", xlabel="Predicted price (USD)", ylabel="Residual (actual − predicted, USD)")
    fig.tight_layout(); fig.savefig(output_dir / "residual_plot.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=40, kde=True, color="#7c3aed", ax=ax)
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.3)
    ax.set(title="Distribution of Prediction Errors", xlabel="Prediction error (actual − predicted, USD)", ylabel="Number of homes")
    fig.tight_layout(); fig.savefig(output_dir / "error_distribution.png", dpi=160); plt.close(fig)

    prices = np.asarray(all_prices, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(prices, bins=45, kde=True, ax=axes[0], color="#f59e0b")
    axes[0].set(title="Original Price Distribution", xlabel="Price (USD)", ylabel="Homes")
    sns.histplot(np.log1p(prices), bins=45, kde=True, ax=axes[1], color="#2563eb")
    axes[1].set(title="After log1p Transformation", xlabel="log1p(price)", ylabel="Homes")
    fig.tight_layout(); fig.savefig(output_dir / "price_distribution.png", dpi=160); plt.close(fig)

    # MLP loss_curve_ is logged-target training loss. MLPRegressor exposes validation R²
    # when early_stopping=True; plot it on a second axis when present.
    try:
        mlp = fitted_model.regressor_.named_steps["mlp"]
        loss = getattr(mlp, "loss_curve_", None)
        scores = getattr(mlp, "validation_scores_", None)
    except (AttributeError, KeyError):
        loss = scores = None
    fig, ax = plt.subplots(figsize=(8, 5))
    if loss is not None and len(loss):
        ax.plot(np.arange(1, len(loss) + 1), loss, color="#2563eb", label="Training loss (log-price scale)")
        ax.set_ylabel("Training loss")
    else:
        ax.text(0.5, 0.5, "The fitted estimator did not expose a loss curve.", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Training loss")
    ax.set_xlabel("Iteration")
    if scores is not None and len(scores):
        right = ax.twinx()
        right.plot(np.arange(1, len(scores) + 1), scores, color="#dc2626", alpha=0.75,
                   label="Internal validation R²")
        right.set_ylabel("Validation R²")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = right.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    elif loss is not None:
        ax.legend(loc="best")
    ax.set_title("MLP Training / Validation Curve")
    fig.tight_layout(); fig.savefig(output_dir / "loss_curve.png", dpi=160); plt.close(fig)


def evaluate_saved_model(bundle: dict, frame, output_dir: Path) -> dict:
    """Re-evaluate the saved model on the reproducible held-out 20% split."""
    from sklearn.model_selection import train_test_split
    from .image_features import extract_image_feature_frame
    from .preprocessing import select_features
    from .data_loader import load_dataset

    frame = load_dataset(frame) if isinstance(frame, (str, Path)) else frame.copy()
    if bundle.get("uses_images"):
        image_dir = Path(bundle.get("image_directory", "socal_pics"))
        if not image_dir.is_absolute():
            image_dir = Path(frame.attrs.get("data_dir", "data")) / image_dir
        img = extract_image_feature_frame(frame, image_dir)
        X = __import__("pandas").concat([select_features(frame), img], axis=1)
    else:
        X = select_features(frame)
    X = X.reindex(columns=bundle["features"])
    y = frame[bundle["target_column"]].astype(float).to_numpy()
    ids = np.arange(len(frame))
    trainval_ids, test_ids = train_test_split(ids, test_size=0.20, random_state=42)
    train_ids, validation_ids = train_test_split(trainval_ids, test_size=0.1875, random_state=42)
    bounds = tuple(bundle["target_bounds_usd"])
    prediction = clip_predictions(bundle["estimator"].predict(X.iloc[test_ids]), bounds)
    validation_prediction = clip_predictions(bundle["estimator"].predict(X.iloc[validation_ids]), bounds)
    refreshed_metrics = {
        "validation_metrics": metric_dict(y[validation_ids], validation_prediction),
        "test_metrics": metric_dict(y[test_ids], prediction),
        "split_rows": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)},
        "features_used": bundle["features"],
        "image_features_used": bool(bundle.get("uses_images")),
        "prediction_bounds_usd": {"minimum": bounds[0], "maximum": bounds[1]},
    }
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
    else:
        metrics = {}
    metrics.update(refreshed_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    create_evaluation_artifacts(y[test_ids], prediction, y, bundle["estimator"], output_dir)
    return metrics
