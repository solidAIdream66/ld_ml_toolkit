from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from .dataset.csv_folder import CSVFolder
from .configs import DataConfig


_DATASET_STATS_CACHE: dict[tuple, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}


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


def _compute_mean_std(
    loader: DataLoader,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:   

    batch_mean = torch.zeros(3)
    batch_mean_sq = torch.zeros(3)

    for images, _ in loader:
        batch_mean += images.mean(dim=(0, 2, 3))
        batch_mean_sq += (images ** 2).mean(dim=(0, 2, 3))

    mean = batch_mean / len(loader)
    var = (batch_mean_sq / len(loader)) - (mean ** 2)
    std = torch.sqrt(var)

    return tuple(mean.tolist()), tuple(std.tolist())


def compute_mean_std(
    data_config: DataConfig,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    root = Path(data_config.data_root)
    preprocess = image_preprocess_transforms(tuple(data_config.image_size))

    train_split = str(data_config.extras.get("train_split", data_config.train_split))
    try:
        train_dir = _resolve_split_dir(root, train_split)
        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=preprocess)
    except FileNotFoundError:
        train_dataset = CSVFolder(data_root=root, transform=preprocess)

    loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    return _compute_mean_std(loader)


def build_train_valid_loaders(
    config: DataConfig,
    use_augmentation: bool = True,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
):
    root = Path(config.data_root)    

    try:
        train_root = _resolve_split_dir(root, config.train_split)
        valid_root = _resolve_split_dir(root, config.valid_split)
        train_dataset = datasets.ImageFolder(
                root=str(train_root),
            )
        valid_dataset = datasets.ImageFolder(
                root=str(valid_root),
            )
        is_subset = False
        stats_cache_key = (
            "imagefolder",
            str(root.resolve()),
            tuple(config.image_size),
            train_root.name.lower(),
            valid_root.name.lower(),
        )
    except FileNotFoundError:
        dataset = CSVFolder(data_root=root)
        train_valid_split = config.extras.get(
            "train_valid_split", config.train_valid_split
        )
        train_size = int(train_valid_split * len(dataset))
        valid_size = len(dataset) - train_size
        # Split into two separate base datasets so transforms don't share state
        train_base = CSVFolder(data_root=root)
        valid_base = CSVFolder(data_root=root)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        train_dataset = torch.utils.data.Subset(train_base, indices[:train_size])
        valid_dataset = torch.utils.data.Subset(valid_base, indices[train_size:])
        is_subset = True
        stats_cache_key = (
            "csvfolder",
            str(root.resolve()),
            tuple(config.image_size),
            float(train_valid_split),
        )

    if mean is None or std is None:
        cached_stats = _DATASET_STATS_CACHE.get(stats_cache_key)
        if cached_stats is not None:
            mean, std = cached_stats
            print(f"Loaded cached mean={mean}, std={std}")
        else:
            preprocess = image_preprocess_transforms(config.image_size)
            if not is_subset:
                train_dataset.transform = preprocess
            else:
                train_dataset.dataset.transform = preprocess
            loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=False,
                    # One-shot scan: never use persistent workers here.
                    # Persistent workers on a temporary loader incur a slow
                    # worker-process shutdown on Windows before the scan finishes.
                    persistent_workers=False,
                )
            mean, std = _compute_mean_std(loader)
            _DATASET_STATS_CACHE[stats_cache_key] = (mean, std)
            print(f"Computed mean={mean}, std={std}")

    train_transform = (
        image_augmented_transforms(config.image_size, mean, std)
        if use_augmentation
        else image_common_transforms(config.image_size, mean, std)
    )
    valid_transform = image_common_transforms(config.image_size, mean, std)

    if is_subset:
        train_dataset.dataset.transform = train_transform
        valid_dataset.dataset.transform = valid_transform
    else:
        train_dataset.transform = train_transform
        valid_dataset.transform = valid_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        # Keep workers alive between epochs: on Windows, re-spawning workers
        # at each validation start adds ~200ms × num_workers of dead time.
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, valid_loader



