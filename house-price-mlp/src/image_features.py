"""Classical, lightweight handcrafted features for the linked SoCal house photos."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from skimage.feature import hog

IMAGE_SIZE = (64, 64)


def feature_names() -> list[str]:
    """Return stable image feature names (RGB stats, simple color ratios, and HOG)."""
    # HOG length for 64x64, 8 orientations, 8x8 cells, 2x2 blocks: 7*7*4*8 = 1,568.
    names = [f"color_{channel}_{stat}" for channel in ("r", "g", "b") for stat in ("mean", "std")]
    names += ["color_brightness", "color_saturation", "color_red_green_ratio", "color_blue_red_ratio"]
    names += [f"hog_{i:04d}" for i in range(1568)]
    return names


def _features_for_pil(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    stats = []
    for channel in range(3):
        plane = rgb[:, :, channel]
        stats.extend((float(plane.mean()), float(plane.std())))
    brightness = float(rgb.mean())
    saturation = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    red, green, blue = (float(rgb[:, :, i].mean()) for i in range(3))
    stats.extend((brightness, saturation, red / (green + 1e-6), blue / (red + 1e-6)))
    descriptor = hog(
        rgb.mean(axis=2), orientations=8, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True,
    ).astype(np.float32)
    return np.concatenate((np.asarray(stats, dtype=np.float32), descriptor))


def extract_image_feature_frame(frame: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Extract 1,578 handcrafted descriptors for each row via exact image_id.jpg linkage."""
    if "image_id" not in frame.columns:
        raise ValueError("The dataset has no image_id column; reliable photo linkage is unavailable.")
    image_dir = Path(image_dir)
    names = feature_names()
    rows = []
    missing = 0
    for value in frame["image_id"]:
        try:
            image_id = int(value)
            path = image_dir / f"{image_id}.jpg"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                rows.append(_features_for_pil(image))
        except (OSError, ValueError, TypeError):
            rows.append(np.full(len(names), np.nan, dtype=np.float32))
            missing += 1
    if missing:
        print(f"Image features unavailable for {missing:,} rows; model imputation will handle them.")
    return pd.DataFrame(np.vstack(rows), columns=names, index=frame.index)


def extract_single_image(image_path: Optional[Path]) -> dict[str, float]:
    """Extract one image's features for CLI prediction, or NaNs if no image is supplied."""
    names = feature_names()
    if image_path is None:
        return {name: float("nan") for name in names}
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        with Image.open(path) as image:
            values = _features_for_pil(image)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read image file {path}: {exc}") from exc
    return dict(zip(names, map(float, values)))
