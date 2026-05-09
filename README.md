# ld_ml_toolkit

A modular ML experimentation framework built on PyTorch Lightning. Easily configure, batch, and track experiments by assembling custom modules and optimizers through structured configurations.

## Overview

`ld_ml_toolkit` streamlines the process of building, training, and evaluating deep learning models for image classification tasks. It provides:

- **Structured Configuration System**: Define experiments via simple JSON/dict configs without code changes
- **Automatic Experiment Tracking**: Built-in logging, checkpointing, and results aggregation
- **PyTorch Lightning Integration**: Modern training loops with mixed precision, distributed training support, and callbacks
- **Flexible Model Architecture**: Support for custom models and torchvision pretrained models
- **Dataset Management**: Multiple dataset formats (CSVFolder, ImageFolder) with automatic augmentation and splitting
- **Kaggle Submission Pipeline**: Generate predictions and submission CSVs for Kaggle competitions

## Key Components

- **ExperimentConfig**: Dataclass-based configuration for data, model, training, and paths
- **Experiment**: Main orchestrator that builds dataloaders, model, optimizer, and runs training via PyTorch Lightning
- **training_pipeline**: Batch runner for executing multiple experiments with parameter sweeps
- **submission_pipeline**: Generate Kaggle submission CSVs from trained models
- **Models**: Pre-built CNN architectures (`vgg_style`, `res_style`) + support for torchvision models

## Configuration Reference

See [CONFIG.md](CONFIG.md) for detailed documentation on all configuration parameters:

- **`data` block**: Dataset settings, image size, batch size, data augmentation
- **`model` block**: Model architecture, number of classes, normalization options
- **`train` block**: Learning rate, epochs, optimizer, scheduler, precision, regularization
- **`paths` block**: Checkpoint and log directories

Configuration uses a two-tier system: `BASE_EXPERIMENT_SETTINGS` (stable defaults) + per-run overrides that are deep-merged before each experiment.

## Quick Start: Training

### 1. Prepare Configuration

Create a JSON config file with your experiment settings:

```python
from ld_ml_toolkit import run_experiments

base_config = {
    "data": {
        "dataset": {"type": "CSVFolder", "root": "path/to/dataset"},
        "image_size": [224, 224],
        "batch_size": 32,
        "num_workers": 4,
    },
    "model": {
        "num_classes": 10,
        "model": {"name": "resnet18", "pretrained": True},
    },
    "train": {
        "epochs": 50,
        "learning_rate": 2e-3,
        "scheduler": {"name": "cosine_warm_restarts", "t0": 10},
    },
}

experiment_list = [
    {"model": {"model": {"name": "resnet18"}}},
    {"model": {"model": {"name": "resnet50"}}},
]
```

### 2. Run Experiments

```python
results = run_experiments(
    experiment_settings_list=experiment_list,
    base_experiment_settings=base_config,
)

# Results logged to:
# - Logs_Checkpoints/Model_checkpoints/version_0/
# - Logs_Checkpoints/Model_logs/version_0/
# - Logs_Checkpoints/experiments.xlsx
```

### 3. Training Details

The `Experiment` class handles:
- Building optimizers (AdamW with no-decay groups for bias/norm)
- Setting up learning rate schedulers (cosine annealing, warm restarts, reduce on plateau)
- PyTorch Lightning training loop with early stopping
- Automatic checkpointing of best validation model
- TensorBoard/CSV logging of metrics

Example experiment result:
```
best_valid_acc=0.9234 | best_valid_loss=0.2456 | epochs_trained=42
```

## Quick Start: Kaggle Submission

### 1. Create submit.py Script

Create a `submit.py` file in your project root:

```python
"""
Submit submission script for generating Kaggle submission CSV using the best model checkpoint.
"""

import os
from pathlib import Path
import argparse

from ld_ml_toolkit import ExperimentConfig, compute_mean_std, load_experiment_config
from ld_ml_toolkit import PathsConfig
from ld_ml_toolkit import ConvClassifier
from ld_ml_toolkit import generate_submission


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV")
    parser.add_argument("--data-root", default="path/to/dataset")
    parser.add_argument("--best-version", required=True)
    parser.add_argument("--experiments-xlsx", default="Logs_Checkpoints/experiments.xlsx")
    parser.add_argument("--test-csv", default="path/to/test.csv")
    parser.add_argument("--output", default="Logs_Checkpoints/submissions/")
    parser.add_argument("--mean", nargs=3, type=float, default=None)
    parser.add_argument("--std", nargs=3, type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_cli_args()

    # Load configuration from experiment tracking sheet
    raw_config = load_experiment_config(
        excel_path=args.experiments_xlsx,
        version_name=args.best_version,
    )
    experiment_cfg = ExperimentConfig.from_settings(raw_config)

    # Compute or use provided normalization stats
    if args.mean is None and args.std is None:
        mean, std = compute_mean_std(data_config=experiment_cfg.data)
        print(f"calculated mean/std from training data: {mean}, {std}")
    elif args.mean is not None and args.std is not None:
        mean = tuple(args.mean)
        std = tuple(args.std)
    else:
        raise ValueError("Provide both --mean and --std together, or omit both to auto-compute")

    # Build model and load checkpoint
    model = ConvClassifier(experiment_cfg.model)
    model_dir = os.path.join(
        PathsConfig.checkpoint_dir,
        args.best_version,
        PathsConfig.model_name
    )

    # Generate submission CSV
    output_path = generate_submission(
        model=model,
        checkpoint_path=model_dir,
        test_csv_path=args.test_csv,
        output_csv_path=os.path.join(args.output, f"submission_{args.best_version}.csv"),
        data_config=experiment_cfg.data,
        mean=mean,
        std=std,
        batch_size=experiment_cfg.data.batch_size,
    )

    print(f"Submission saved to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()
```

### 2. Run the Submission Script

```bash
python submit.py \
    --best-version version_0 \
    --experiments-xlsx Logs_Checkpoints/experiments.xlsx \
    --test-csv path/to/test.csv \
    --output Logs_Checkpoints/submissions/
```

**Arguments:**
- `--best-version`: Model checkpoint version (e.g., `version_0`, `version_42`)
- `--experiments-xlsx`: Path to experiment tracking spreadsheet
- `--test-csv`: Path to test dataset CSV with image IDs
- `--mean`, `--std`: Optional: Custom normalization stats (auto-computed from training data if omitted)
- `--output`: Directory for output submission CSV

### 3. Output

Generates `submission_{version}.csv` with columns:
```
id,class
image_001.jpg,0
image_002.jpg,1
...
```

Ready to upload to Kaggle!

### 4. Normalization

By default, the script computes mean/std from the training dataset. Provide explicit values to override:

```bash
python submit.py \
    --best-version version_0 \
    --test-csv path/to/test.csv \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225
```

## Supported Models

**Custom Models:**
- `vgg_style`: VGG-inspired architecture with configurable depth/width
- `res_style`: ResNet-inspired architecture with residual blocks

**Torchvision Models** (e.g., `resnet18`, `resnet50`, `efficientnet_b0`, `vgg16`, etc.):
```python
"model": {"name": "efficientnet_b4", "pretrained": True}
```

See [CONFIG.md](CONFIG.md) for the complete list of 100+ supported torchvision models.

## Dataset Formats

### CSVFolder (Recommended)

```
dataset/
  images/
    image_1.jpg
    image_2.jpg
    ...
  train.csv        # columns: id, class
  test.csv         # columns: id
```

Config:
```python
"data": {
    "dataset": {
        "type": "CSVFolder",
        "root": "dataset",
        "train_valid_split": 0.8,
    }
}
```

### ImageFolder

```
dataset/
  Train/
    class_0/
      image_1.jpg
    class_1/
  Valid/
    class_0/
    class_1/
  Test/
```

Config:
```python
"data": {
    "dataset": {
        "type": "ImageFolder",
        "root": "dataset",
        "train_split": "Train",
        "valid_split": "Valid",
        "test_split": "Test",
    }
}
```

## Installation

```bash
pip install torch pytorch-lightning torchmetrics torchvision pandas openpyxl
```

If the toolkit is in a local `ld_library` folder:
```bash
pip install -e /path/to/ld_library/ld_ml_toolkit
```

## License

See [LICENSE](LICENSE) file.
