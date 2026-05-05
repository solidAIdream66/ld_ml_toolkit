import json
import re
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

from .configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    TrainConfig,
)


def experiment_config_to_dict(experiment_config: ExperimentConfig) -> Dict[str, Any]:
    return asdict(experiment_config)


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def _known_field_names(dataclass_type) -> Iterable[str]:
    return {f.name for f in fields(dataclass_type)}


def _split_known_unknown(
    section_data: Dict[str, Any], dataclass_type
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    known_keys = _known_field_names(dataclass_type)
    known = {k: v for k, v in section_data.items() if k in known_keys}
    unknown = {k: v for k, v in section_data.items() if k not in known_keys}
    return known, unknown


def experiment_config_from_dict(config_data: Dict[str, Any]) -> ExperimentConfig:
    payload = dict(config_data or {})
    _normalize_train_config_payload(payload)

    data_known, data_unknown = _split_known_unknown(
        dict(payload.get("data", {})), DataConfig
    )
    model_known, model_unknown = _split_known_unknown(
        dict(payload.get("model", {})), ModelConfig
    )
    train_known, train_unknown = _split_known_unknown(
        dict(payload.get("train", {})), TrainConfig
    )
    paths_known, paths_unknown = _split_known_unknown(
        dict(payload.get("paths", {})), PathsConfig
    )

    extras = _normalize_extras(dict(payload.get("extras", {})))
    for key in ("data", "model", "train", "paths", "extras"):
        payload.pop(key, None)

    # Keep unknown/new settings so older code can still round-trip future configs.
    extras.update({f"data.{k}": v for k, v in data_unknown.items()})
    extras.update({f"model.{k}": v for k, v in model_unknown.items()})
    extras.update({f"train.{k}": v for k, v in train_unknown.items()})
    extras.update({f"paths.{k}": v for k, v in paths_unknown.items()})
    extras.update(payload)

    return ExperimentConfig(
        data=DataConfig(**data_known),
        model=ModelConfig(**model_known),
        train=TrainConfig(**train_known),
        paths=PathsConfig(**paths_known),
        extras=extras,
    )


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None

    try:
        if text.startswith("0") and text != "0" and not text.startswith("0."):
            raise ValueError
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    is_json_like = (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    )
    if is_json_like:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    return value


def _set_nested_value(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = data
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def _normalize_extras(extras: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(extras)

    scheduler = dict(normalized.get("scheduler", {}))
    scheduler_key_map = {
        "train.scheduler_t0": "t0",
        "train.scheduler_t_mult": "t_mult",
        "train.scheduler_eta_min": "eta_min",
        "train.scheduler_factor": "factor",
        "train.scheduler_patience": "patience",
    }
    for old_key, new_key in scheduler_key_map.items():
        if old_key in normalized and new_key not in scheduler:
            scheduler[new_key] = normalized.pop(old_key)
    if scheduler:
        normalized["scheduler"] = scheduler

    dataloader = dict(normalized.get("dataloader", {}))
    if "train.use_augmentation" in normalized and "use_augmentation" not in dataloader:
        dataloader["use_augmentation"] = normalized.pop("train.use_augmentation")
    if dataloader:
        normalized["dataloader"] = dataloader

    return normalized


def _normalize_train_config_payload(payload: Dict[str, Any]) -> None:
    train_section = dict(payload.get("train", {}))
    model_section = dict(payload.get("model", {}))
    extras_section = dict(payload.get("extras", {}))

    # Migrate renamed fields from old configs so they round-trip correctly.
    if "scheduler_type" in train_section and "scheduler" not in train_section:
        train_section["scheduler"] = train_section.pop("scheduler_type")
    if "model_name" in model_section and "model" not in model_section:
        model_section["model"] = model_section.pop("model_name")
    # Migrate old train.extras.scheduler dict → scheduler_params field.
    if "scheduler" in extras_section and "scheduler_params" not in train_section:
        train_section["scheduler_params"] = extras_section.pop("scheduler")

    single_key_mappings = {
        ("optimizer", "name"): "optimizer_type",
        ("loss", "name"): "loss_type",
        ("metric", "name"): "metric_type",
        ("visualization", "logger"): "logger_type",
        ("dataloader", "use_augmentation"): "use_augmentation",
    }

    for (section_name, source_key), train_key in single_key_mappings.items():
        section = extras_section.get(section_name)
        if train_key in train_section:
            continue
        if isinstance(section, dict) and source_key in section:
            train_section[train_key] = section[source_key]

    for section_name in (
        "optimizer", "loss", "metric", "visualization", "dataloader"
    ):
        section = extras_section.get(section_name)
        if isinstance(section, dict):
            stripped_section = {
                key: value
                for key, value in section.items()
                if key not in {"name", "logger", "use_augmentation"}
            }
            if stripped_section:
                extras_section[section_name] = stripped_section
            else:
                extras_section.pop(section_name, None)

    payload["train"] = train_section
    payload["model"] = model_section
    payload["extras"] = extras_section


def apply_overrides(
    config_data: Dict[str, Any],
    overrides: Optional[Iterable[str]],
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(config_data))
    if not overrides:
        return updated

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}."
                " Expected format key=value"
            )
        dotted_key, raw_value = override.split("=", 1)
        dotted_key = dotted_key.strip()
        if not dotted_key:
            raise ValueError(f"Invalid override '{override}'. Key cannot be empty")
        _set_nested_value(updated, dotted_key, _parse_scalar(raw_value))

    return updated


def _next_version_name(root_log_dir: Path) -> str:
    if not root_log_dir.exists():
        return "version_0"

    version_numbers = []
    for child in root_log_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^version_(\d+)$", child.name)
        if match:
            version_numbers.append(int(match.group(1)))

    if not version_numbers:
        return "version_0"
    return f"version_{max(version_numbers) + 1}"


def setup_log_directory(
    path_config: PathsConfig,
    version_name: Optional[str] = None,
) -> Tuple[PathsConfig, str]:
    """Create per-run versioned log/checkpoint folders."""
    root_log_dir = Path(path_config.log_dir)
    root_checkpoint_dir = Path(path_config.checkpoint_dir)

    resolved_version_name = version_name or _next_version_name(root_log_dir)
    log_dir = root_log_dir / resolved_version_name
    checkpoint_dir = root_checkpoint_dir / resolved_version_name

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    updated_paths = PathsConfig(
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        model_name=path_config.model_name,
    )
    return updated_paths, resolved_version_name


def append_experiment_record(
    excel_path: Path,
    config_data: Dict[str, Any],
    metrics_data: Dict[str, Any],
    version_name: str,
) -> Dict[str, Any]:
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "version_name": version_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_json": json.dumps(config_data, sort_keys=True),
        "metrics_json": json.dumps(metrics_data, sort_keys=True),
    }

    record.update(
        {f"cfg.{k}": v for k, v in _flatten_dict(config_data).items()}
    )
    record.update(
        {f"metric.{k}": v for k, v in _flatten_dict(metrics_data).items()}
    )

    row_df = pd.DataFrame([record])

    if excel_path.exists():
        existing_df = pd.read_excel(excel_path)
        existing_df = existing_df.drop(columns=["run_id", "run_name"], errors="ignore")
        out_df = pd.concat([existing_df, row_df], ignore_index=True)
    else:
        out_df = row_df

    out_df.to_excel(excel_path, index=False)
    return record


def load_experiment_config(
    excel_path: Path,
    version_name: Optional[str] = None,
    row_index: Optional[int] = None,
) -> Dict[str, Any]:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if df.empty:
        raise ValueError(f"Experiment file is empty: {excel_path}")

    if version_name:
        if "version_name" not in df.columns:
            raise ValueError(f"Column 'version_name' not found in {excel_path}")
        matches = df[df["version_name"].astype(str) == str(version_name)]
        if matches.empty:
            raise ValueError(
                f"version_name {version_name!r} not found"
                f" in {excel_path}"
            )
        row = matches.iloc[-1]
    elif row_index is not None:
        row = df.iloc[row_index]
    else:
        row = df.iloc[-1]

    config_json = row.get("config_json")
    if isinstance(config_json, str) and config_json.strip():
        return json.loads(config_json)

    # Fallback for older files with only flattened columns.
    cfg_columns = [col for col in df.columns if col.startswith("cfg.")]
    config_data: Dict[str, Any] = {}
    for col in cfg_columns:
        value = row[col]
        if pd.isna(value):
            continue
        _set_nested_value(config_data, col[4:], value)

    if not config_data:
        raise ValueError(f"No config information found in {excel_path}")
    return config_data
