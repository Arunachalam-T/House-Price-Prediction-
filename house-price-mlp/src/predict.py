"""Schema-driven interactive price prediction for a saved MLP model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd

from .image_features import extract_single_image
from .prediction_utils import clip_predictions

LABELS = {
    "bed": "Bedrooms",
    "bath": "Bathrooms",
    "sqft": "Square Feet",
    "citi": "City",
}


def _field_label(column: str) -> str:
    return LABELS.get(column.lower(), column.replace("_", " ").strip().title())


def _ask_yes_no(prompt: str, *, default: bool = False, input_fn=input,
                output_fn=print) -> bool:
    while True:
        try:
            answer = input_fn(prompt).strip().lower()
        except EOFError:
            return False
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        output_fn("Please enter y or n.")


def _numeric_prompt(column: str, series: pd.Series, input_fn=input,
                    output_fn=print):
    label = _field_label(column)
    is_integer = pd.api.types.is_integer_dtype(series.dtype)
    low = float(pd.to_numeric(series, errors="coerce").min())
    high = float(pd.to_numeric(series, errors="coerce").max())
    kind = "integer" if is_integer else "number"
    while True:
        try:
            raw = input_fn(f"{label} ({kind}, observed range {low:g}–{high:g}): ").strip()
        except EOFError:
            raise
        try:
            value = int(raw) if is_integer else float(raw)
            if not math.isfinite(float(value)):
                raise ValueError("Value must be finite.")
            if not low <= float(value) <= high:
                raise ValueError(f"Enter a value from {low:g} to {high:g}.")
            return value
        except ValueError as exc:
            output_fn(f"Invalid {label.lower()}: {exc}")


def _categorical_prompt(column: str, series: pd.Series, input_fn=input,
                        output_fn=print):
    label = _field_label(column)
    choices = series.dropna().astype(str).value_counts().head(8).index.tolist()
    if choices:
        output_fn(f"Common {label.lower()} choices:")
        for index, choice in enumerate(choices, start=1):
            output_fn(f"  {index}. {choice}")
    while True:
        prompt = f"{label} (enter a name or choice number): " if choices else f"{label}: "
        try:
            raw = input_fn(prompt).strip()
        except EOFError:
            raise
        if not raw:
            output_fn(f"{label} cannot be blank.")
            continue
        if choices and raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
            output_fn(f"Choose a number from 1 to {len(choices)}, or type a {label.lower()} name.")
            continue
        # Unknown categories are allowed; the saved OneHotEncoder ignores unseen values.
        return raw


def predict_price(bundle: dict, house_features: dict[str, object],
                  image_path: Path | None = None) -> float:
    """Predict using exactly the saved bundle's feature list and preprocessing pipeline."""
    features = list(bundle["features"])
    image_columns = set(bundle.get("image_features", []))
    missing = [name for name in features if name not in image_columns and name not in house_features]
    if missing:
        raise ValueError(f"Missing required input feature(s): {', '.join(missing)}")
    row = dict(house_features)
    if bundle.get("uses_images"):
        row.update(extract_single_image(image_path))
    sample = pd.DataFrame([row]).reindex(columns=features)
    bounds = tuple(bundle["target_bounds_usd"])
    prediction = bundle["estimator"].predict(sample)
    return float(clip_predictions(prediction, bounds)[0])


def interactive_prediction_loop(bundle: dict, reference_frame: pd.DataFrame,
                                input_fn=input, output_fn=print,
                                fixed_image_path: Path | None = None) -> None:
    """Ask for the exact saved model inputs and permit repeated house predictions."""
    image_columns = set(bundle.get("image_features", []))
    model_features = [name for name in bundle["features"] if name not in image_columns]
    missing = [name for name in model_features if name not in reference_frame.columns]
    if missing:
        raise ValueError(
            "The dataset does not contain all features required by the saved model: "
            + ", ".join(missing)
        )
    numeric = [name for name in model_features
               if pd.api.types.is_numeric_dtype(reference_frame[name].dtype)]
    categorical = [name for name in model_features if name not in numeric]
    ordered_features = numeric + categorical

    while True:
        output_fn("\n========== HOUSE PRICE PREDICTION ==========")
        output_fn("Enter house details:")
        try:
            values = {}
            for column in ordered_features:
                if column in numeric:
                    values[column] = _numeric_prompt(column, reference_frame[column], input_fn, output_fn)
                else:
                    values[column] = _categorical_prompt(column, reference_frame[column], input_fn, output_fn)

            image_path = fixed_image_path
            if bundle.get("uses_images") and image_path is None:
                while True:
                    try:
                        raw_image = input_fn("Optional exterior image path (press Enter to skip): ").strip()
                    except EOFError:
                        raise
                    if not raw_image:
                        break
                    candidate = Path(raw_image).expanduser()
                    if candidate.is_file():
                        image_path = candidate
                        break
                    output_fn(f"Image file not found: {candidate}. Try again or press Enter to skip.")

            price = predict_price(bundle, values, image_path=image_path)
        except EOFError:
            output_fn("\nNo more interactive input; exiting prediction mode.")
            return
        except (ValueError, FileNotFoundError, OSError) as exc:
            output_fn(f"Could not make this prediction: {exc}")
            continue

        output_fn("=============================================")
        output_fn(f"Predicted House Price: ${price:,.2f}")
        output_fn("=============================================")
        if not _ask_yes_no("Predict another house? (y/n): ", input_fn=input_fn, output_fn=output_fn):
            output_fn("Goodbye.")
            return


def interactive_predict(model_path: Path, image_path: Path | None = None,
                        input_fn=input, reference_frame: pd.DataFrame | None = None,
                        output_fn=print) -> None:
    """Compatibility wrapper for optional --predict developer mode."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run `python main.py` to train it.")
    bundle = joblib.load(model_path)
    if reference_frame is None:
        raise ValueError("A reference dataset is required to infer model input types. Use the main app or pass reference_frame.")
    interactive_prediction_loop(bundle, reference_frame, input_fn=input_fn,
                                output_fn=output_fn, fixed_image_path=image_path)
