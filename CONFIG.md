# Experiment Configuration Reference

Settings are split into a `BASE_EXPERIMENT_SETTINGS` dict (stable defaults) and a list of per-run override dicts (`EXPERIMENT_SETTINGS_LIST`). Both are deep-merged before each run.

---

## `data` block

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_root` | str | `"dataset"` | Path to the dataset root. Supports two folder structures (see below). |
| `image_size` | [int, int] | `[224, 224]` | Resize target `[height, width]`. |
| `batch_size` | int | `32` | Mini-batch size for both train and validation loaders. |
| `num_workers` | int | `2` | DataLoader worker processes. Set to `0` on Windows if workers crash. For GPU training, increase this to keep augmentation and image decoding from starving the device. |
| `pin_memory` | bool | `True` | Pin memory for faster GPU transfer. |
| `test_split` | str | `"Test"` | Sub-folder name used for the test set (case-insensitive resolution). |
| `extras` | dict | `{}` | Extended options — see below. |

### Supported dataset structures

**Structure A — ImageFolder** (two explicit split directories)

```
data_root/
    Train/          # or any name matching train_split
        class_a/
        class_b/
    Valid/          # or any name matching valid_split
        class_a/
        class_b/
    Test/
        ...
```

Set inside `data.extras`:

```python
"extras": {
    "train_split": "Train",   # sub-folder name for training images
    "valid_split": "Valid",   # sub-folder name for validation images
}
```

When both `train_split` and `valid_split` directories are found, `torchvision.datasets.ImageFolder` is used and `train_valid_split` is ignored.

---

**Structure B — CSVFolder** (single image pool + CSV manifest)

```
data_root/
    images/
        images/
            image_1.jpg
            image_2.jpg
    train.csv     # columns: id, class
    test.csv
```

Set inside `data.extras`:

```python
"extras": {
    "train_valid_split": 0.8,   # fraction of train.csv rows used for training
}
```

When no split directories are found, `CSVFolder` is used and the dataset is randomly split according to `train_valid_split`. The random split uses a fixed seed (42) for reproducibility.

---

## `model` block

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | str | `"vgg_style"` | Architecture. Must be a key in `extras["model"]`. Supported: `"vgg_style"`, `"res_style"`. |
| `num_classes` | int | `3` | Number of output classes. |
| `use_batch_norm` | bool | `True` | Insert BatchNorm layers after each convolution. |
| `dropout` | float | `0.0` | Dropout probability applied inside residual blocks (res_style) or the classifier head (vgg_style). |
| `channels` | List[int] \| None | `None` | Output channels per stage. When `None`, filled from the model's entry in `extras["model"]`. |
| `blocks_per_stage` | List[int] \| None | `None` | Conv/residual blocks per stage (always a list). When `None`, filled from the model's entry in `extras["model"]`. |
| `extras` | dict | lookup table | Registry of valid model names and their default params. See below. |

### `model` lookup table (`extras["model"]`)

Each entry maps a model name to its default `channels` and `blocks_per_stage`.

```python
"extras": {
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
```

`channels` and `blocks_per_stage` can be set directly alongside `model`; they override the lookup defaults:

```python
"model": {
    "model": "res_style",
    "channels": [32, 64, 128, 256],       # optional override
    "blocks_per_stage": [2, 2, 3, 2],     # optional override
}
```

---

## `train` block

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `epochs` | int | `50` | Total training epochs. |
| `learning_rate` | float | `2e-3` | Initial learning rate for AdamW. |
| `weight_decay` | float | `1e-4` | L2 penalty applied to weight tensors (bias/norm params are exempt). |
| `label_smoothing` | float | `0.0` | Label smoothing factor for the training loss. Validation loss is always unsmoothed. |
| `device` | str | `"cuda"` | Accelerator: `"cuda"` or `"cpu"`. |
| `seed` | int | `21` | Global random seed for reproducibility. |
| `optimizer_type` | str | `"adamw"` | Optimizer. Currently only `"adamw"` is supported. |
| `loss_type` | str | `"cross_entropy"` | Loss function. Currently only `"cross_entropy"` is supported. |
| `metric_type` | str | `"accuracy"` | Evaluation metric. Currently only `"accuracy"` is supported. |
| `logger_type` | str | `"tensorboard"` | Experiment logger. Currently only `"tensorboard"` is supported. |
| `use_augmentation` | bool | `True` | Apply random augmentation to the training set (rotation, crop, flip, color jitter). |
| `scheduler` | str | `"cosine_warm_restarts"` | LR scheduler. Must be a key in `extras["scheduler"]` or `"none"`/`"off"`/`"disabled"`. Supported: `"cosine_warm_restarts"`, `"plateau"`, and their aliases. |
| `precision` | str | `"32"` | Trainer precision. Supported: `"32"`, `"16-mixed"`, `"bf16-mixed"`. Mixed precision is usually the best speed default on CUDA. |
| `deterministic` | bool | `True` | When `True`, prefer reproducible kernels. Set `False` during sweeps to enable faster cuDNN benchmarking on fixed-size image training. |
| `scheduler_params` | dict \| None | `None` | Active scheduler parameters. When `None`, filled from `extras["scheduler"][scheduler]`. Partial dicts are merged with defaults (user values win). |
| `matmul_precision` | str | `"high"` | PyTorch matmul precision: `"highest"`, `"high"`, or `"medium"`. |
| `extras` | dict | lookup table | Registry of valid scheduler names and their default params. See below. |

### `scheduler` lookup table and `scheduler_params`

`extras["scheduler"]` is the registry. `scheduler_params` holds the resolved active params after `__post_init__` merges.

#### `"cosine_warm_restarts"` (aliases: `"cosine"`, `"cosineannealingwarmrestarts"`)

Default `scheduler_params`:

```python
"scheduler_params": {
    "t0":      10,    # (int)   epochs in the first restart cycle
    "t_mult":   2,    # (int)   cycle length multiplier after each restart
    "eta_min": 1e-5,  # (float) minimum learning rate
}
```

#### `"plateau"` (aliases: `"reduce_on_plateau"`, `"reducelronplateau"`)

Default `scheduler_params`:

```python
"scheduler_params": {
    "factor":   0.5,  # (float) multiplicative factor on plateau
    "patience": 3,    # (int)   epochs with no improvement before reducing LR
}
```

#### `"none"` (aliases: `"off"`, `"disabled"`)

Disables the scheduler. `scheduler_params` will be `{}`.

To override specific params without setting all of them:

```python
"train": {
    "scheduler": "cosine_warm_restarts",
    "scheduler_params": {"t0": 20},   # only t0 overridden; t_mult and eta_min use defaults
}
```

---

## `extras` (top-level)

| Key | Type | Description |
|-----|------|-------------|
| `exp_name` | str | Human-readable label recorded in `experiments.xlsx`. Defaults to the auto-generated version folder name if omitted. |

---

## Minimal override example

Only specify keys that differ from the base; the rest are deep-merged from `BASE_EXPERIMENT_SETTINGS`:

```python
EXPERIMENT_SETTINGS_LIST = [
    {
        "train": {
            "seed": 3407,
            "weight_decay": 1e-4,
        },
        "extras": {"exp_name": "wd_1e4_seed3407"},
    },
]
```
