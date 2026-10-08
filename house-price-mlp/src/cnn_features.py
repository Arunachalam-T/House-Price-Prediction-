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
        # Load a pretrained ResNet18 model as specified in the multimodal project abstract
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
        return features.cpu().numpy().flatten()

def extract_image_feature_frame(frame: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Extract CNN features from images using ResNet18."""
    print(f"Extracting CNN features for {len(frame)} images... This may take a while.")
    extractor = CNNFeatureExtractor()
    image_dir = Path(image_dir)
    names = [f"cnn_{i:04d}" for i in range(512)]
    rows = []
    missing = 0
    
    try:
        from tqdm import tqdm
        iterator = tqdm(frame["image_id"], desc="Extracting Image Features")
    except ImportError:
        iterator = frame["image_id"]

    for value in iterator:
        try:
            image_id = int(value)
            path = image_dir / f"{image_id}.jpg"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                rows.append(extractor.extract(image))
        except (OSError, ValueError, TypeError, FileNotFoundError):
            rows.append(np.full(len(names), np.nan, dtype=np.float32))
            missing += 1
            
    if missing:
        print(f"Image features unavailable for {missing:,} rows; model imputation will handle them.")
    return pd.DataFrame(np.vstack(rows), columns=names, index=frame.index)


