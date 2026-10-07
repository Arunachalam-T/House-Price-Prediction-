"""Prediction safety helpers for bounded house-price estimates."""
from __future__ import annotations

import numpy as np


def clip_predictions(values, bounds: tuple[float, float]) -> np.ndarray:
    """Replace non-finite predictions and keep prices within the training target range."""
    lower, upper = map(float, bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper < lower:
        raise ValueError(f"Invalid training price bounds: {bounds}")
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=lower, posinf=upper, neginf=lower)
    return np.clip(values, lower, upper)
