# Experiment Configuration Reference

Settings are split into a `BASE_EXPERIMENT_SETTINGS` dict (stable defaults) and a list of per-run override dicts (`EXPERIMENT_SETTINGS_LIST`). Both are deep-merged before each run.

---

## `data` block

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dataset` | dict | `{ "type": "CSVFolder", "root": "dataset", "train_valid_split": 0.8 }` | Dataset settings grouped in one nested block (see structure-specific keys below). |
| `image_size` | [int, int] | `[224, 224]` | Resize target `[height, width]`. |
| `batch_size` | int | `32` | Mini-batch size for both train and validation loaders. |
| `num_workers` | int | `2` | DataLoader worker processes. Set to `0` on Windows if workers crash. For GPU training, increase this to keep augmentation and image decoding from starving the device. |
| `pin_memory` | bool | `True` | Pin memory for faster GPU transfer. |
| `extras` | dict | `{}` | Extended options — see below. |

### `data.dataset` common keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `"CSVFolder"` | Dataset loader mode. Use `"CSVFolder"` or `"ImageFolder"`. |
| `root` | str | `"dataset"` | Path to dataset root directory. |

### Supported dataset structures

**Structure A — ImageFolder** (two explicit split directories)

```
dataset.root/
    Train/          # or any name matching train_split
        class_a/
        class_b/
    Valid/          # or any name matching valid_split
        class_a/
        class_b/
    Test/
        ...
```

Set inside `data.dataset`:

```python
"dataset": {
    "type": "ImageFolder",
    "root": "dataset",
    "train_split": "Train",   # sub-folder name for training images
    "valid_split": "Valid",   # sub-folder name for validation images
    "test_split": "Test",     # sub-folder name for test images
}
```

When `type` is `ImageFolder`, train/valid data is loaded from the configured split folders under `root`.

---

**Structure B — CSVFolder** (single image pool + CSV manifest)

```
dataset.root/
    images/
        images/
            image_1.jpg
            image_2.jpg
    train.csv     # columns: id, class
    test.csv
```

Set inside `data.dataset`:

```python
"dataset": {
    "type": "CSVFolder",
    "root": "dataset",
    "train_valid_split": 0.8,   # fraction of train.csv rows used for training
}
```

When `type` is `CSVFolder`, `CSVFolder` is used and the dataset is randomly split according to `train_valid_split`. The random split uses a fixed seed (42) for reproducibility.

---

## `model` block

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | Dict[str, Any] | `{ "name": "vgg_style" }` | Required model spec. Must include `name` plus optional kwargs, for example `{ "name": "resnet18", "pretrained": true }` or `{ "name": "res_style", "channels": [...], "blocks_per_stage": [...] }`. |
| `num_classes` | int | `3` | Number of output classes. |
| `use_batch_norm` | bool | `True` | Insert BatchNorm layers after each convolution. |
| `dropout` | float | `0.0` | Dropout probability applied inside residual blocks (res_style) or the classifier head (vgg_style). |
| `channels` | List[int] \| None | `None` | Output channels per stage. If omitted, can be provided under `model.<name>.channels` for custom models. |
| `blocks_per_stage` | List[int] \| None | `None` | Conv/residual blocks per stage. If omitted, can be provided under `model.<name>.blocks_per_stage` for custom models. |
| `extras` | dict | `{}` | Optional custom metadata. |

### `model.model.name` supported values

Custom toolkit models:

- `res_style`
- `vgg_style`

Torchvision models:

`model.model.name` also accepts any value from `torchvision.models.list_models()` in the active environment.

Current list (torchvision 0.20.1):

- `alexnet`
- `convnext_base`
- `convnext_large`
- `convnext_small`
- `convnext_tiny`
- `deeplabv3_mobilenet_v3_large`
- `deeplabv3_resnet101`
- `deeplabv3_resnet50`
- `densenet121`
- `densenet161`
- `densenet169`
- `densenet201`
- `efficientnet_b0`
- `efficientnet_b1`
- `efficientnet_b2`
- `efficientnet_b3`
- `efficientnet_b4`
- `efficientnet_b5`
- `efficientnet_b6`
- `efficientnet_b7`
- `efficientnet_v2_l`
- `efficientnet_v2_m`
- `efficientnet_v2_s`
- `fasterrcnn_mobilenet_v3_large_320_fpn`
- `fasterrcnn_mobilenet_v3_large_fpn`
- `fasterrcnn_resnet50_fpn`
- `fasterrcnn_resnet50_fpn_v2`
- `fcn_resnet101`
- `fcn_resnet50`
- `fcos_resnet50_fpn`
- `googlenet`
- `inception_v3`
- `keypointrcnn_resnet50_fpn`
- `lraspp_mobilenet_v3_large`
- `maskrcnn_resnet50_fpn`
- `maskrcnn_resnet50_fpn_v2`
- `maxvit_t`
- `mc3_18`
- `mnasnet0_5`
- `mnasnet0_75`
- `mnasnet1_0`
- `mnasnet1_3`
- `mobilenet_v2`
- `mobilenet_v3_large`
- `mobilenet_v3_small`
- `mvit_v1_b`
- `mvit_v2_s`
- `quantized_googlenet`
- `quantized_inception_v3`
- `quantized_mobilenet_v2`
- `quantized_mobilenet_v3_large`
- `quantized_resnet18`
- `quantized_resnet50`
- `quantized_resnext101_32x8d`
- `quantized_resnext101_64x4d`
- `quantized_shufflenet_v2_x0_5`
- `quantized_shufflenet_v2_x1_0`
- `quantized_shufflenet_v2_x1_5`
- `quantized_shufflenet_v2_x2_0`
- `r2plus1d_18`
- `r3d_18`
- `raft_large`
- `raft_small`
- `regnet_x_16gf`
- `regnet_x_1_6gf`
- `regnet_x_32gf`
- `regnet_x_3_2gf`
- `regnet_x_400mf`
- `regnet_x_800mf`
- `regnet_x_8gf`
- `regnet_y_128gf`
- `regnet_y_16gf`
- `regnet_y_1_6gf`
- `regnet_y_32gf`
- `regnet_y_3_2gf`
- `regnet_y_400mf`
- `regnet_y_800mf`
- `regnet_y_8gf`
- `resnet101`
- `resnet152`
- `resnet18`
- `resnet34`
- `resnet50`
- `resnext101_32x8d`
- `resnext101_64x4d`
- `resnext50_32x4d`
- `retinanet_resnet50_fpn`
- `retinanet_resnet50_fpn_v2`
- `s3d`
- `shufflenet_v2_x0_5`
- `shufflenet_v2_x1_0`
- `shufflenet_v2_x1_5`
- `shufflenet_v2_x2_0`
- `squeezenet1_0`
- `squeezenet1_1`
- `ssd300_vgg16`
- `ssdlite320_mobilenet_v3_large`
- `swin3d_b`
- `swin3d_s`
- `swin3d_t`
- `swin_b`
- `swin_s`
- `swin_t`
- `swin_v2_b`
- `swin_v2_s`
- `swin_v2_t`
- `vgg11`
- `vgg11_bn`
- `vgg13`
- `vgg13_bn`
- `vgg16`
- `vgg16_bn`
- `vgg19`
- `vgg19_bn`
- `vit_b_16`
- `vit_b_32`
- `vit_h_14`
- `vit_l_16`
- `vit_l_32`
- `wide_resnet101_2`
- `wide_resnet50_2`

Note: this toolkit's classifier-head replacement expects classification-style models with an `fc` or `classifier` linear head. Detection, segmentation, flow, and keypoint models from torchvision are listed for completeness but may not be directly usable with `ConvClassifier`.

### `model` examples

```python
"model": {
    "model": {
        "name": "res_style",
        "channels": [64, 128, 256, 512],
        "blocks_per_stage": [2, 2, 2, 2],
    },
    "num_classes": 13,
    "use_batch_norm": True,
    "dropout": 0.1,
}
```

```python
"model": {
    "model": {
        "name": "resnet18",
        "pretrained": True,
    },
    "num_classes": 13,
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
| `scheduler` | Dict[str, Any] | `{ "name": "cosine_warm_restarts", "t0": 10, "t_mult": 2, "eta_min": 1e-5 }` | LR scheduler config. Use `scheduler.name` to pick scheduler type and set parameters in the same dict. |
| `precision` | str | `"32"` | Trainer precision. Supported: `"32"`, `"16-mixed"`, `"bf16-mixed"`. Mixed precision is usually the best speed default on CUDA. |
| `deterministic` | bool | `True` | When `True`, prefer reproducible kernels. Set `False` during sweeps to enable faster cuDNN benchmarking on fixed-size image training. |
| `matmul_precision` | str | `"high"` | PyTorch matmul precision: `"highest"`, `"high"`, or `"medium"`. |
| `extras` | dict | lookup table | Registry of valid scheduler names and their default params. See below. |

### `train.scheduler.name` supported values

- `cosine_warm_restarts`
- `cosine` (alias of cosine_warm_restarts)
- `cosineannealingwarmrestarts` (alias of cosine_warm_restarts)
- `plateau`
- `reduce_on_plateau` (alias of plateau)
- `reducelronplateau` (alias of plateau)
- `none`
- `off` (alias of none)
- `disabled` (alias of none)

### `scheduler` lookup table

`extras["scheduler"]` is the registry. The active values are merged directly into `train.scheduler`.

#### `"cosine_warm_restarts"` (aliases: `"cosine"`, `"cosineannealingwarmrestarts"`)

Default `scheduler`:

```python
"scheduler": {
    "name":   "cosine_warm_restarts",
    "t0":      10,    # (int)   epochs in the first restart cycle
    "t_mult":   2,    # (int)   cycle length multiplier after each restart
    "eta_min": 1e-5,  # (float) minimum learning rate
}
```

#### `"plateau"` (aliases: `"reduce_on_plateau"`, `"reducelronplateau"`)

Default `scheduler`:

```python
"scheduler": {
    "name":    "plateau",
    "factor":   0.5,  # (float) multiplicative factor on plateau
    "patience": 3,    # (int)   epochs with no improvement before reducing LR
}
```

#### `"none"` (aliases: `"off"`, `"disabled"`)

Disables the scheduler.

To override specific params without setting all of them:

```python
"train": {
    "scheduler": {
        "name": "cosine_warm_restarts",
        "t0": 20,   # only t0 overridden; t_mult and eta_min use defaults
    },
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
