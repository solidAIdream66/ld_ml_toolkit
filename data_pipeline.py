from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

from .configs import DataConfig


def _resolve_split_dir(root: Path, split_name: str) -> Path:
    direct = root / split_name
    if direct.exists():
        return direct

    lower_map = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    candidate = lower_map.get(split_name.lower())
    if candidate:
        return candidate

    raise FileNotFoundError(f"Split '{split_name}' not found under {root}")


def image_preprocess_transforms(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


def image_common_transforms(
    image_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def image_augmented_transforms(
    image_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> v2.Compose:
    return v2.Compose([
        v2.RandomRotation(degrees=15),
        v2.RandomResizedCrop(size=image_size),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])


def create_loader(
    root: Path,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = datasets.ImageFolder(root=str(root), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def compute_mean_std(
    train_root: Path,
    image_size: Tuple[int, int],
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    loader = create_loader(
        root=train_root,
        transform=image_preprocess_transforms(image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    batch_mean = torch.zeros(3)
    batch_mean_sq = torch.zeros(3)

    for images, _ in loader:
        batch_mean += images.mean(dim=(0, 2, 3))
        batch_mean_sq += (images ** 2).mean(dim=(0, 2, 3))

    mean = batch_mean / len(loader)
    var = (batch_mean_sq / len(loader)) - (mean ** 2)
    std = torch.sqrt(var)

    return tuple(mean.tolist()), tuple(std.tolist())


def build_train_valid_loaders(
    config: DataConfig,
    use_augmentation: bool = True,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
):
    root = Path(config.data_root)
    train_root = _resolve_split_dir(root, config.train_split)
    valid_root = _resolve_split_dir(root, config.valid_split)

    if mean is None or std is None:
        mean, std = compute_mean_std(
            train_root=train_root,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

    train_transform = (
        image_augmented_transforms(config.image_size, mean, std)
        if use_augmentation
        else image_common_transforms(config.image_size, mean, std)
    )
    valid_transform = image_common_transforms(config.image_size, mean, std)

    train_loader = create_loader(
        root=train_root,
        transform=train_transform,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True,
    )
    valid_loader = create_loader(
        root=valid_root,
        transform=valid_transform,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True,
    )

    return train_loader, valid_loader, mean, std


def get_class_names(config: DataConfig):
    root = Path(config.data_root)
    train_root = _resolve_split_dir(root, config.train_split)
    return datasets.ImageFolder(root=str(train_root)).classes
