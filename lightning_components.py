from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from .configs import PathsConfig, TrainConfig

import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, train_loader, valid_loader):
        super().__init__()
        self._train_loader = train_loader
        self._valid_loader = valid_loader

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._valid_loader
    


class ClassifierModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        train_loss_fn: nn.Module,
        valid_loss_fn: nn.Module,
        train_metric: nn.Module,
        valid_metric: nn.Module,
        lr_scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loss_fn = train_loss_fn
        self.valid_loss_fn = valid_loss_fn
        self.train_metric = train_metric
        self.valid_metric = valid_metric
        self.lr_scheduler_config = lr_scheduler_config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.train_loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        self.train_metric.update(preds, targets)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )
        self.log("train/acc", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.valid_loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        self.valid_metric.update(preds, targets)

        self.log(
            "valid/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )
        self.log("valid/acc", self.valid_metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.lr_scheduler_config is None:
            return self.optimizer

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler_config,
        }


class HistoryCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
        }

    @staticmethod
    def _to_float(metric_value) -> float:
        if metric_value is None:
            return float("nan")
        if isinstance(metric_value, torch.Tensor):
            return float(metric_value.detach().cpu().item())
        return float(metric_value)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Skip the pre-training sanity validation pass so history aligns with epoch count.
        if getattr(trainer, "sanity_checking", False):
            return

        metrics = trainer.callback_metrics
        self.history["train_loss"].append(self._to_float(metrics.get("train/loss")))
        self.history["train_acc"].append(self._to_float(metrics.get("train/acc")))
        self.history["valid_loss"].append(self._to_float(metrics.get("valid/loss")))
        self.history["valid_acc"].append(self._to_float(metrics.get("valid/acc")))

        # Also log an epoch-indexed view for TensorBoard plots.
        # logger = trainer.logger
        # experiment = getattr(logger, "experiment", None)
        # if experiment is not None and hasattr(experiment, "add_scalar"):
        #     epoch_idx = trainer.current_epoch + 1
        #     train_loss = self._to_float(metrics.get("train/loss"))
        #     train_acc = self._to_float(metrics.get("train/acc"))
        #     valid_loss = self._to_float(metrics.get("valid/loss"))
        #     valid_acc = self._to_float(metrics.get("valid/acc"))

        #     experiment.add_scalar("epoch/train_loss", train_loss, epoch_idx)
        #     experiment.add_scalar("epoch/train_acc", train_acc, epoch_idx)
        #     experiment.add_scalar("epoch/valid_loss", valid_loss, epoch_idx)
        #     experiment.add_scalar("epoch/valid_acc", valid_acc, epoch_idx)


def resolve_accelerator_and_devices(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def export_best_state_dict(
    model: nn.Module,
    best_checkpoint_path: str,
    path_config: PathsConfig,
) -> Path:
    checkpoint_dir = Path(path_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output_state_dict_path = checkpoint_dir / path_config.model_name
    checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    cleaned_state_dict = {
        key.replace("model.", "", 1) if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)
    torch.save(model.state_dict(), output_state_dict_path)
    return output_state_dict_path
