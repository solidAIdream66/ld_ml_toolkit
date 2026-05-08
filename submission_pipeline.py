from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

from .configs import DataConfig
from .data_pipeline import image_common_transforms


class TestImageDataset(Dataset):
    def __init__(self, image_ids: Sequence[str], image_root: Union[str, Path], transform):
        self.image_ids = [str(image_id) for image_id in image_ids]
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(self.image_root / f"{image_id}.jpg").convert("RGB")
        return self.transform(image), image_id


def _resolve_image_root(root: Path) -> Path:
    csv_style_root = root / "images" / "images"
    if csv_style_root.exists():
        return csv_style_root

    return root


def _build_idx_to_class(root: Path) -> Dict[int, str]:
    train_csv = root / "train.csv"
    train_df = pd.read_csv(train_csv)
    classes = sorted(train_df["class"].astype(str).unique().tolist())
    return {idx: cls_name for idx, cls_name in enumerate(classes)}


def generate_submission(
    model,
    checkpoint_path: Union[str, Path],
    test_csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
    data_config: DataConfig,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    batch_size: int = 64,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    test_csv_path = Path(test_csv_path)
    output_csv_path = Path(output_csv_path)

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    root = Path(data_config.data_root)
    test_root = _resolve_image_root(root)
    idx_to_class = _build_idx_to_class(root)

    test_df = pd.read_csv(test_csv_path)
    test_id_col = "id" if "id" in test_df.columns else "ID"
    transform = image_common_transforms(data_config.image_size, mean, std)
    dataset = TestImageDataset(test_df[test_id_col].tolist(), test_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    predictions = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().tolist()
            predictions.extend(idx_to_class[pred] for pred in preds)

    submission_df = pd.DataFrame({
        "id": test_df[test_id_col].astype(str),
        "class": predictions,
    })
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_csv_path, index=False)
    return output_csv_path
