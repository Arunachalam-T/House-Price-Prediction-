"""Standalone entrypoint: run ``python main.py`` for the complete application."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.data_loader import ensure_csv, ensure_images, load_dataset
from src.evaluate import evaluate_saved_model
from src.predict import interactive_prediction_loop, interactive_predict
from src.train import train_model

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "trained_model.joblib"
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUTS = ROOT / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="House Price Prediction using sklearn MLPRegressor. Run without flags for the full application."
    )
    action = parser.add_mutually_exclusive_group(required=False)
    action.add_argument("--train", action="store_true", help="Developer mode: tune and train the MLP, then exit")
    action.add_argument("--evaluate", action="store_true", help="Developer mode: evaluate the saved model, then exit")
    action.add_argument("--predict", action="store_true", help="Developer mode: enter prediction mode using the saved model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset directory (default: ./data)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Saved model path")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS, help="Metrics/plots output directory")
    parser.add_argument("--use-images", action="store_true", help="Developer training option: add handcrafted linked-photo features")
    parser.add_argument("--image", type=Path, help="Optional exterior photo for --predict with an image-trained model")
    parser.add_argument("--search-iterations", type=int, default=4, help="RandomizedSearchCV candidates (default: 4; maximum 8)")
    parser.add_argument("--cv", type=int, default=3, help="Training-fold CV count (default: 3)")
    return parser


def _print_metrics(metrics: dict) -> None:
    test = metrics.get("test_metrics", {})
    output = metrics.get("outputs_dir")
    print("\n========== MODEL EVALUATION (USD) ==========")
    if "mae_usd" in test:
        print(f"MAE : ${test['mae_usd']:,.2f}")
    if "rmse_usd" in test:
        print(f"RMSE: ${test['rmse_usd']:,.2f}")
    if "r2" in test:
        print(f"R²  : {test['r2']:.4f}")
    if output:
        print(f"Full metrics and plots saved under: {output}")
    print("=============================================")


def _cache_is_compatible(bundle: dict, frame) -> bool:
    if not isinstance(bundle, dict) or "estimator" not in bundle or "features" not in bundle:
        return False
    if bundle.get("target_column") not in frame.columns:
        return False
    image_features = set(bundle.get("image_features", []))
    return all(name in frame.columns for name in bundle["features"] if name not in image_features)


def _run_full_application(args: argparse.Namespace) -> None:
    print("House Price Prediction using MLPRegressor")
    print("Loading dataset...")
    csv_path = ensure_csv(args.data)
    frame = load_dataset(csv_path)
    frame.attrs["data_dir"] = str(args.data)
    print(f"Loaded {len(frame):,} houses from {csv_path.name}.")

    bundle = None
    metrics = None
    if args.model.is_file():
        print("Loading existing trained model...")
        try:
            candidate = joblib.load(args.model)
            if _cache_is_compatible(candidate, frame):
                bundle = candidate
            else:
                print("The saved model does not match this dataset schema; training a new model.")
        except Exception as exc:
            print(f"Could not load the saved model ({exc}); training a new model.")

    if bundle is not None:
        from src.predict import _ask_yes_no
        if _ask_yes_no("Retrain model? (y/n): ", default=False):
            bundle = None

    if bundle is None:
        print("Preprocessing, splitting, and training MLPRegressor. This may take a few minutes...")
        metrics = train_model(
            args.data, args.model, args.outputs, use_images=args.use_images,
            n_iter=args.search_iterations, cv=args.cv,
        )
        bundle = joblib.load(args.model)
    else:
        if bundle.get("uses_images"):
            ensure_images(args.data)
        print("Evaluating the saved model on the reproducible validation/test split...")
        metrics = evaluate_saved_model(bundle, frame, args.outputs)

    metrics["outputs_dir"] = str(args.outputs)
    _print_metrics(metrics)
    print("\nStarting interactive prediction mode. The model is loaded/trained once for this session.")
    interactive_prediction_loop(bundle, frame)


def main() -> None:
    args = build_parser().parse_args()
    if args.train:
        train_model(args.data, args.model, args.outputs, use_images=args.use_images,
                    n_iter=args.search_iterations, cv=args.cv)
    elif args.evaluate:
        if not args.model.is_file():
            raise FileNotFoundError(f"Saved model not found: {args.model}. Run `python main.py` first.")
        bundle = joblib.load(args.model)
        csv_path = ensure_csv(args.data)
        frame = load_dataset(csv_path)
        frame.attrs["data_dir"] = str(args.data)
        metrics = evaluate_saved_model(bundle, frame, args.outputs)
        _print_metrics(metrics)
    elif args.predict:
        csv_path = ensure_csv(args.data)
        frame = load_dataset(csv_path)
        interactive_predict(args.model, image_path=args.image, reference_frame=frame)
    else:
        _run_full_application(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication cancelled.")
    except EOFError:
        print("\nInput stream ended; exiting cleanly.")
