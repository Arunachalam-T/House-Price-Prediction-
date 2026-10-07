"""Dataset discovery and optional public Kaggle download helpers."""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

KAGGLE_DATASET = "ted8080/house-prices-and-images-socal"
KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/" + KAGGLE_DATASET
)


def find_csv(data_dir: Path) -> Optional[Path]:
    """Find the expected CSV first, then any CSV in the provided data directory."""
    data_dir = Path(data_dir)
    preferred = [data_dir / "socal2.csv", data_dir / "socal2" / "socal2.csv"]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted(data_dir.rglob("*.csv")) if data_dir.exists() else []
    return candidates[0] if candidates else None


def _download_archive(destination: Path) -> Path:
    """Download the public Kaggle archive in streaming mode; no credentials required."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(KAGGLE_DOWNLOAD_URL, stream=True, timeout=(30, 180))
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0) or 0)
    received = 0
    with destination.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
                received += len(chunk)
                if total and received % (32 * 1024 * 1024) < len(chunk):
                    print(f"Downloaded {received / (1024**2):.0f} / {total / (1024**2):.0f} MB")
    if total and received != total:
        destination.unlink(missing_ok=True)
        raise IOError(f"Incomplete Kaggle download: expected {total} bytes, got {received}.")
    return destination


def ensure_csv(data_dir: Path) -> Path:
    """Return the local CSV or obtain the expected CSV from Kaggle if absent."""
    data_dir = Path(data_dir)
    existing = find_csv(data_dir)
    if existing:
        return existing
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="socal-dataset-") as temp_dir:
        archive = _download_archive(Path(temp_dir) / "dataset.zip")
        with zipfile.ZipFile(archive) as zipped:
            member = next((n for n in zipped.namelist() if Path(n).name.lower() == "socal2.csv"), None)
            if member is None:
                raise FileNotFoundError("The Kaggle archive did not contain socal2.csv.")
            destination = data_dir / "socal2.csv"
            with zipped.open(member) as source, destination.open("wb") as output:
                output.write(source.read())
    return destination


def ensure_images(data_dir: Path) -> Path:
    """Ensure image_id-named JPEGs are in data/socal_pics; download archive if needed."""
    data_dir = Path(data_dir)
    image_dir = data_dir / "socal_pics"
    if image_dir.is_dir() and any(image_dir.glob("*.jpg")):
        return image_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="socal-images-") as temp_dir:
        archive = _download_archive(Path(temp_dir) / "dataset.zip")
        image_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                normalized = member.filename.replace("\\", "/")
                if not normalized.startswith("socal2/socal_pics/") or member.is_dir():
                    continue
                basename = Path(normalized).name
                if not basename.lower().endswith(".jpg"):
                    continue
                # Use only the filename, avoiding extraction of untrusted paths.
                with zipped.open(member) as source, (image_dir / basename).open("wb") as output:
                    output.write(source.read())
                count += 1
                if count % 2000 == 0:
                    print(f"Extracted {count:,} house images...")
        if count == 0:
            raise FileNotFoundError("No images were found under socal2/socal_pics in the Kaggle archive.")
    print(f"Extracted {count:,} images to {image_dir}")
    return image_dir


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read and validate the dataset's observed price target column."""
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    target = next((column for column in frame.columns if column.lower() == "price"), None)
    if target is None:
        raise ValueError(
            f"Could not find a 'price' target column in {csv_path}. "
            f"Observed columns: {frame.columns.tolist()}"
        )
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    before = len(frame)
    frame = frame.loc[frame[target].notna() & (frame[target] > 0)].copy()
    removed = before - len(frame)
    if removed:
        print(f"Removed {removed} rows with missing or non-positive target price.")
    if frame.empty:
        raise ValueError("No rows with a valid positive price remain after validation.")
    return frame.reset_index(drop=True)
