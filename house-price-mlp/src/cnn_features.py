import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

class CNNFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pretrained ResNet18 as per project specifications
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer to get 512-dimensional features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten().astype(np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        tensors = [self.transform(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
        return features.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)


def feature_names() -> list[str]:
    return [f"cnn_{i:04d}" for i in range(512)]


def extract_single_image(image_path: Path | None = None) -> dict[str, float]:
    """Extract 512 CNN features for a single input image during prediction."""
    names = feature_names()
    if image_path is None:
        return {name: float("nan") for name in names}
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        extractor = CNNFeatureExtractor()
        with Image.open(path) as image:
            values = extractor.extract(image)
        return dict(zip(names, map(float, values)))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read image file {path}: {exc}") from exc


def extract_image_feature_frame(frame: pd.DataFrame, image_dir: Path, cache_path: Path | None = None) -> pd.DataFrame:
    """Extract CNN features from images using ResNet18, with caching and batching."""
    names = feature_names()
    if cache_path is not None and Path(cache_path).is_file():
        print(f"Loading cached CNN features from {cache_path}...")
        try:
            cached = pd.read_csv(cache_path, index_col=0)
            if len(cached) == len(frame) and all(c in cached.columns for c in names):
                cached.index = frame.index
                return cached[names]
        except Exception as exc:
            print(f"Cache read error ({exc}); re-extracting features...")

    print(f"Extracting ResNet-18 CNN features for {len(frame)} images...")
    extractor = CNNFeatureExtractor()
    image_dir = Path(image_dir)
    batch_size = 64 if extractor.device.type == "cuda" else 16
    results = [None] * len(frame)

    # Process in batches
    batch_images = []
    batch_indices = []

    for idx, value in enumerate(frame["image_id"]):
        try:
            image_id = int(value)
            path = image_dir / f"{image_id}.jpg"
            if not path.is_file():
                results[idx] = np.full(512, np.nan, dtype=np.float32)
                continue
            with Image.open(path) as img:
                batch_images.append(img.copy())
            batch_indices.append(idx)
        except Exception:
            results[idx] = np.full(512, np.nan, dtype=np.float32)
            continue

        if len(batch_images) >= batch_size:
            feats = extractor.extract_batch(batch_images)
            for i, feat in zip(batch_indices, feats):
                results[i] = feat
            batch_images.clear()
            batch_indices.clear()
            if (idx + 1) % 1024 == 0 or (idx + 1) == len(frame):
                print(f"Processed {idx + 1}/{len(frame)} images...")

    if batch_images:
        feats = extractor.extract_batch(batch_images)
        for i, feat in zip(batch_indices, feats):
            results[i] = feat
        batch_images.clear()
        batch_indices.clear()

    rows = [r if r is not None else np.full(512, np.nan, dtype=np.float32) for r in results]
    df_features = pd.DataFrame(np.vstack(rows), columns=names, index=frame.index)

    if cache_path is not None:
        try:
            df_features.to_csv(cache_path)
            print(f"Saved CNN features to cache: {cache_path}")
        except Exception as exc:
            print(f"Could not save cache to {cache_path}: {exc}")

    return df_features



