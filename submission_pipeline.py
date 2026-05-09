from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple, Union

import pandas as pd
from PIL import Image
from torchvision.datasets.folder import find_classes as find_imagefolder_classes

import torch
from torch.utils.data import DataLoader, Dataset

from .configs import DataConfig
from .data_pipeline import image_common_transforms
from .dataset.csv_folder import find_classes as find_csvfolder_classes


class TestImageDataset(Dataset):
    def __init__(
        self,
        image_ids: Sequence[str],
        image_root: Union[str, Path],
        transform: Callable,
    ):
        self.image_ids = [str(image_id) for image_id in image_ids]
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, image_id: str) -> Path:
        image_rel_path = Path(image_id)
        image_path = self.image_root / image_rel_path
        if image_path.suffix:
            return image_path
        return image_path.with_suffix(".jpg")

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = self._resolve_image_path(image_id)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_id


def _resolve_image_root(root: Path) -> Path:
    csv_style_root = root / "images" / "images"
    if csv_style_root.exists():
        return csv_style_root

    return root


def _resolve_split_dir(root: Path, split_name: str) -> Path:
    direct = root / split_name
    if direct.exists():
        return direct

    lower_map = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    candidate = lower_map.get(split_name.lower())
    if candidate:
        return candidate

    raise FileNotFoundError(f"Split '{split_name}' not found under {root}")


def _build_idx_to_class(root: Path, dataset_cfg: Dict[str, object]) -> Dict[int, str]:
    dataset_type = str(dataset_cfg.get("type", "CSVFolder")).strip().lower()

    if dataset_type in {"csvfolder", "cvsfolder"}:
        classes, class_to_idx = find_csvfolder_classes(root / "train.csv")
    elif dataset_type == "imagefolder":
        train_split = str(dataset_cfg.get("train_split", "Train"))
        train_root = _resolve_split_dir(root, train_split)
        classes, class_to_idx = find_imagefolder_classes(str(train_root))
    else:
        raise ValueError(f"Unsupported dataset type for submission: {dataset_type}")

    return {idx: cls_name for cls_name, idx in class_to_idx.items()}


def _resolve_test_root(root: Path, dataset_cfg: Dict[str, object]) -> Path:
    dataset_type = str(dataset_cfg.get("type", "CSVFolder")).strip().lower()
    if dataset_type in {"csvfolder", "cvsfolder"}:
        return _resolve_image_root(root)
    if dataset_type == "imagefolder":
        test_split = str(dataset_cfg.get("test_split", "Test"))
        return _resolve_split_dir(root, test_split)
    raise ValueError(f"Unsupported dataset type for submission: {dataset_type}")


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

    dataset_cfg = dict(data_config.dataset or {})
    root = Path(str(dataset_cfg.get("root", "dataset")))
    test_root = _resolve_test_root(root, dataset_cfg)
    idx_to_class = _build_idx_to_class(root, dataset_cfg)

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
