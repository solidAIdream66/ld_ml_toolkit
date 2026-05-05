from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    data_root: str = "dataset"
    train_split: str = "Train"
    valid_split: str = "Valid"
    test_split: str = "Test"
    train_valid_split: float = 0.8
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    num_classes: int = 3
    model: str = "vgg_style"
    use_batch_norm: bool = True
    dropout: float = 0
    channels: Optional[List[int]] = None
    blocks_per_stage: Optional[List[int]] = None
    extras: Dict[str, Dict] = field(
        default_factory=lambda: {
            "model": {
                "vgg_style": {
                    "channels": [64, 128, 256, 512, 512],
                    "blocks_per_stage": [2, 2, 2, 2, 2],
                },
                "res_style": {
                    "channels": [64, 128, 256, 512],
                    "blocks_per_stage": [3, 4, 6, 3],
                },
            }
        }
    )

    def __post_init__(self) -> None:
        """Validate model and fill in defaults for channels/blocks_per_stage."""
        valid_models = self.extras.get("model", {})
        if self.model not in valid_models:
            raise ValueError(
                f"model={self.model!r} is not supported. "
                f"Valid values: {sorted(valid_models.keys())}"
            )
        defaults = valid_models[self.model]
        if self.channels is None:
            self.channels = list(defaults["channels"])
        if self.blocks_per_stage is None:
            self.blocks_per_stage = list(defaults["blocks_per_stage"])


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
    scheduler: str = "cosine_warm_restarts"
    precision: str = "32"
    deterministic: bool = True
    matmul_precision: str = "high"
    scheduler_params: Optional[Dict[str, Any]] = None
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
        """Validate scheduler and fill in defaults for scheduler_params."""
        stype = self.scheduler.strip().lower()
        valid_schedulers = self.extras.get("scheduler", {})
        precision = str(self.precision).strip().lower()
        valid_precisions = {"32", "16-mixed", "bf16-mixed"}

        # Allow "none", "off", "disabled" as special values
        if stype not in {*valid_schedulers, "none", "off", "disabled"}:
            raise ValueError(
                f"scheduler={self.scheduler!r} is not supported. "
                f"Valid values: {sorted(valid_schedulers.keys())} or 'none'/'off'/'disabled'"
            )
        if precision not in valid_precisions:
            raise ValueError(
                f"precision={self.precision!r} is not supported. "
                f"Valid values: {sorted(valid_precisions)}"
            )

        self.precision = precision
        defaults = dict(valid_schedulers.get(stype, {}))
        if self.scheduler_params is None:
            self.scheduler_params = defaults
        else:
            # Merge user-supplied params with defaults (user params take precedence)
            self.scheduler_params = {**defaults, **self.scheduler_params}


@dataclass
class PathsConfig:
    checkpoint_dir: str = "Logs_Checkpoints/Model_checkpoints"
    log_dir: str = "Logs_Checkpoints/Model_logs"
    model_name: str = "cat_dog_panda_classifier.pt"
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
