import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    dataset: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "CSVFolder",
            "root": "dataset",
            "train_valid_split": 0.8,
        }
    )
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    num_classes: int = 3
    model: Dict[str, Any] = field(default_factory=lambda: {"name": "vgg_style"})
    use_batch_norm: bool = True
    dropout: float = 0
    channels: Optional[List[int]] = None
    blocks_per_stage: Optional[List[int]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model spec shape and lift common custom-model kwargs."""
        if not isinstance(self.model, dict) or not self.model:
            raise ValueError(
                "model must be a non-empty dict like {'name': 'resnet18', 'pretrained': True}"
            )

        raw_name = self.model.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("model.name must be a non-empty string")

        if self.channels is None and "channels" in self.model:
            self.channels = list(self.model["channels"])
        if self.blocks_per_stage is None and "blocks_per_stage" in self.model:
            self.blocks_per_stage = list(self.model["blocks_per_stage"])

        normalized = dict(self.model)
        normalized["name"] = raw_name.strip().lower()
        self.model = normalized


@dataclass
class TrainConfig:
    epochs: int = 50
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    device: str = "cuda"
    seed: int = 21
    optimizer_type: str = "adamw"
    loss_type: str = "cross_entropy"
    metric_type: str = "accuracy"
    logger_type: str = "tensorboard"
    use_augmentation: bool = True
    scheduler: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "cosine_warm_restarts",
            "t0": 10,
            "t_mult": 2,
            "eta_min": 1e-5,
        }
    )
    precision: str = "32"
    deterministic: bool = True
    matmul_precision: str = "high"
    extras: Dict[str, Dict] = field(
        default_factory=lambda: {
            "scheduler": {
                "cosine_warm_restarts": {"t0": 10, "t_mult": 2, "eta_min": 1e-5},
                "cosine": {"t0": 10, "t_mult": 2, "eta_min": 1e-5},
                "cosineannealingwarmrestarts": {"t0": 10, "t_mult": 2, "eta_min": 1e-5},
                "plateau": {"factor": 0.5, "patience": 3},
                "reduce_on_plateau": {"factor": 0.5, "patience": 3},
                "reducelronplateau": {"factor": 0.5, "patience": 3},
            }
        }
    )

    def __post_init__(self) -> None:
        """Validate scheduler config and normalize defaults."""
        if not isinstance(self.scheduler, dict):
            raise ValueError(
                "scheduler must be a dict like {'name': 'cosine_warm_restarts', 't0': 10, 't_mult': 2, 'eta_min': 1e-5}"
            )

        scheduler_cfg = dict(self.scheduler)
        stype = str(scheduler_cfg.get("name", "")).strip().lower()
        valid_schedulers = self.extras.get("scheduler", {})
        precision = str(self.precision).strip().lower()
        valid_precisions = {"32", "16-mixed", "bf16-mixed"}

        # Allow "none", "off", "disabled" as special values
        if stype not in {*valid_schedulers, "none", "off", "disabled"}:
            raise ValueError(
                f"scheduler.name={scheduler_cfg.get('name')!r} is not supported. "
                f"Valid values: {sorted(valid_schedulers.keys())} or 'none'/'off'/'disabled'"
            )
        if precision not in valid_precisions:
            raise ValueError(
                f"precision={self.precision!r} is not supported. "
                f"Valid values: {sorted(valid_precisions)}"
            )

        self.precision = precision
        defaults = dict(valid_schedulers.get(stype, {}))
        self.scheduler = {"name": stype, **defaults, **scheduler_cfg}


@dataclass
class PathsConfig:
    checkpoint_dir: str = "Logs_Checkpoints/Model_checkpoints"
    log_dir: str = "Logs_Checkpoints/Model_logs"
    model_name: str = "classifier.pt"
    experiments_file: str = "Logs_Checkpoints/experiments.xlsx"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def get_value_from_dict(
        config_dict: Dict[str, Any],
        dotted_key: str,
        default: Any = None,
    ) -> Any:
        """Get a nested value from a config dict using dotted-key access."""
        if not dotted_key:
            return default

        cursor: Any = config_dict
        for part in dotted_key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]

        return cursor

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a nested value using dotted-key access."""
        return self.get_value_from_dict(
            self.to_dict(), dotted_key, default
        )

    @staticmethod
    def deep_merge_dict(
        base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge override into a deep copy of base."""
        merged = deepcopy(base)
        for key, value in override.items():
            both_dicts = (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            )
            if both_dicts:
                merged[key] = ExperimentConfig.deep_merge_dict(
                    merged[key], value
                )
            else:
                merged[key] = deepcopy(value)
        return merged

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "ExperimentConfig":
        """Build an ExperimentConfig by merging settings over defaults."""
        from .experiment_tracking import experiment_config_from_dict

        default_config = cls().to_dict()
        merged_config = cls.deep_merge_dict(default_config, settings)
        return experiment_config_from_dict(merged_config)


def _upgrade_legacy_data_section(config_data: Dict[str, Any]) -> Dict[str, Any]:
    upgraded = deepcopy(config_data)
    data_section = dict(upgraded.get("data", {}))
    dataset_section = dict(data_section.get("dataset", {}))

    # Promote legacy top-level data keys to the nested data.dataset structure.
    if "data_root" in data_section and "root" not in dataset_section:
        dataset_section["root"] = data_section["data_root"]
    if "train_valid_split" in data_section and "train_valid_split" not in dataset_section:
        dataset_section["train_valid_split"] = data_section["train_valid_split"]
    if "train_split" in data_section and "train_split" not in dataset_section:
        dataset_section["train_split"] = data_section["train_split"]
    if "valid_split" in data_section and "valid_split" not in dataset_section:
        dataset_section["valid_split"] = data_section["valid_split"]
    if "test_split" in data_section and "test_split" not in dataset_section:
        dataset_section["test_split"] = data_section["test_split"]

    if dataset_section and "type" not in dataset_section:
        dataset_section["type"] = "CSVFolder" if "train_valid_split" in dataset_section else "ImageFolder"

    if dataset_section:
        data_section["dataset"] = dataset_section

    for key in ("data_root", "train_valid_split", "train_split", "valid_split", "test_split"):
        data_section.pop(key, None)

    upgraded["data"] = data_section
    return upgraded


def get_config(dataset: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment settings from a JSON file and normalize to toolkit schema.

    The returned dict matches the same nested shape as BASE_EXPERIMENT_SETTINGS,
    with defaults filled in for omitted sections/keys.

        dataset can be either:
        - A direct path to a .json file.
        - A dataset key/name (e.g. "13-kenyan-foods"), resolved as
            <config_dir>/<name>.json, where config_dir defaults to ld_ml_toolkit/configs.
    """
    
    dataset_path = Path(dataset)
    if dataset_path.suffix.lower() == ".json" or dataset_path.parent != Path("."):
        config_path = dataset_path
    else:
        base_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parent / "configs"
        config_path = base_dir / f"{dataset}.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        raw_config = json.load(fp)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Config JSON must be an object with sections like data/model/train/paths"
        )

    upgraded = _upgrade_legacy_data_section(raw_config)
    return ExperimentConfig.from_settings(upgraded).to_dict()


def upload_config(
    experiments_xlsx: str = "Logs_Checkpoints/experiments.xlsx",
    dataset_name: Optional[str] = None,
    config_dir: Optional[str] = None,
) -> Path:
    """Export the best experiment config to a readable JSON file.

    The output file is created under ld_ml_toolkit/configs by default and named
    with the dataset key (for example: 13-kenyan-foods.json).
    """
    import pandas as pd

    excel_path = Path(experiments_xlsx)
    if not excel_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if df.empty:
        raise ValueError(f"Experiment file is empty: {excel_path}")

    if "metric.best_valid_acc" not in df.columns:
        raise ValueError(
            "Column 'metric.best_valid_acc' is required to select the best experiment"
        )

    sortable = df.copy()
    sortable["metric.best_valid_acc"] = pd.to_numeric(
        sortable["metric.best_valid_acc"], errors="coerce"
    )
    if "metric.best_valid_loss" in sortable.columns:
        sortable["metric.best_valid_loss"] = pd.to_numeric(
            sortable["metric.best_valid_loss"], errors="coerce"
        )
    else:
        sortable["metric.best_valid_loss"] = float("inf")

    sortable = sortable.dropna(subset=["metric.best_valid_acc"])
    if sortable.empty:
        raise ValueError("No valid rows found for metric.best_valid_acc")

    best_row = sortable.sort_values(
        by=["metric.best_valid_acc", "metric.best_valid_loss"],
        ascending=[False, True],
    ).iloc[0]

    raw_config = best_row.get("config_json")
    if isinstance(raw_config, dict):
        config_data = raw_config
    elif isinstance(raw_config, str) and raw_config.strip():
        config_data = json.loads(raw_config)
    else:
        cfg_columns = [col for col in sortable.columns if col.startswith("cfg.")]
        if not cfg_columns:
            raise ValueError("No config_json or cfg.* columns found in experiment file")

        config_data = {}
        for col in cfg_columns:
            value = best_row[col]
            if pd.isna(value):
                continue

            keys = col[4:].split(".")
            cursor = config_data
            for key in keys[:-1]:
                child = cursor.get(key)
                if not isinstance(child, dict):
                    child = {}
                    cursor[key] = child
                cursor = child
            cursor[keys[-1]] = value

    upgraded_config = _upgrade_legacy_data_section(config_data)
    normalized = ExperimentConfig.from_settings(upgraded_config).to_dict()

    resolved_dataset_name = dataset_name
    if not resolved_dataset_name:
        root_value = str(normalized.get("data", {}).get("dataset", {}).get("root", ""))
        if not root_value:
            root_value = str(config_data.get("data", {}).get("data_root", ""))
        extras_root = str(config_data.get("extras", {}).get("data.data_root", ""))
        if (not root_value or root_value == "dataset") and extras_root:
            root_value = extras_root
        if not root_value:
            root_value = "dataset"
        resolved_dataset_name = Path(root_value).name or "dataset"

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved_dataset_name).strip("-._")
    if not safe_name:
        safe_name = "dataset"

    output_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parent / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name}.json"

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(normalized, fp, indent=2)
        fp.write("\n")

    return output_path
