from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim import AdamW, Optimizer, lr_scheduler
from torchmetrics.classification import MulticlassAccuracy

from .configs import ExperimentConfig
from .data_pipeline import build_train_valid_loaders
from .experiment_tracking import append_experiment_record
from .lightning_components import (
    ClassifierModule,
    DataModule,
    HistoryCallback,
    export_best_state_dict,
    resolve_accelerator_and_devices,
)
from .models import ConvClassifier


class Experiment:
    """Assemble and run one experiment from configuration."""

    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config

        self.train_loader, self.valid_loader, self.mean, self.std = self._build_dataloaders()
        self.data_module = DataModule(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
        )

        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()
        self.train_metric, self.valid_metric = self._build_metrics()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler_config = self._build_scheduler_config()
        self.logger = self._build_logger()
        self.history_callback = HistoryCallback()
        self.checkpoint_callback = self._build_checkpoint_callback()
        self.lightning_module = self._build_lightning_module()

    def _extras_section(self, section_name: str) -> Dict:
        section = self.config.extras.get(section_name, {})
        if isinstance(section, dict):
            return section
        return {}

    def _build_dataloaders(self):
        return build_train_valid_loaders(
            self.config.data,
            use_augmentation=bool(self.config.train.use_augmentation),
        )

    def _build_model(self) -> nn.Module:
        return ConvClassifier(self.config.model)

    def _build_optimizer(self) -> Optimizer:
        optimizer_name = str(
            self.config.train.optimizer_type
        ).strip().lower()
        # Extend here to support more optimizer types (e.g. "sgd", "adam").
        if optimizer_name != "adamw":
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}"
            )

        return AdamW(
            self.model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )

    def _build_loss_fn(self) -> nn.Module:
        loss_name = str(self.config.train.loss_type).strip().lower()
        # Extend here to support more loss types (e.g. "focal").
        if loss_name != "cross_entropy":
            raise ValueError(f"Unsupported loss: {loss_name}")

        return nn.CrossEntropyLoss(
            label_smoothing=self.config.train.label_smoothing
        )

    def _build_metrics(self) -> Tuple[nn.Module, nn.Module]:
        metric_name = str(self.config.train.metric_type).strip().lower()
        # Extend here to support more metric types (e.g. "f1").
        if metric_name != "accuracy":
            raise ValueError(f"Unsupported metric: {metric_name}")

        num_classes = self.config.model.num_classes
        return (
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassAccuracy(num_classes=num_classes),
        )

    def _build_scheduler_config(self):
        raw = self.config.train.scheduler_type or "none"
        scheduler_type = raw.strip().lower()
        if scheduler_type in {"none", "off", "disabled"}:
            return None

        scheduler_cfg = self._extras_section("scheduler")

        # Extend here to support more scheduler types.
        # Each branch must return a Lightning lr_scheduler config dict
        # with at least "scheduler" and "interval" keys.
        # ReduceLROnPlateau also requires a "monitor" key.
        cosine_aliases = {
            "cosine", "cosine_warm_restarts", "cosineannealingwarmrestarts"
        }
        if scheduler_type in cosine_aliases:
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(scheduler_cfg.get("t0", 10)),
                T_mult=int(scheduler_cfg.get("t_mult", 2)),
                eta_min=float(scheduler_cfg.get("eta_min", 1e-5)),
            )
            return {"scheduler": scheduler, "interval": "epoch"}

        plateau_aliases = {
            "plateau", "reduce_on_plateau", "reducelronplateau"
        }
        if scheduler_type in plateau_aliases:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 3)),
            )
            return {
                "scheduler": scheduler,
                "monitor": "valid/loss",
                "interval": "epoch",
            }

        raise ValueError(f"Unsupported scheduler_type: {raw!r}")

    def _build_logger(self):
        logger_name = str(self.config.train.logger_type).strip().lower()
        # Extend here to support more logger types (e.g. "wandb", "csv").
        if logger_name != "tensorboard":
            raise ValueError(f"Unsupported logger: {logger_name}")

        logger_root = Path(self.config.paths.log_dir)
        logger_root.mkdir(parents=True, exist_ok=True)
        return TensorBoardLogger(
            save_dir=str(logger_root), name="", version=""
        )

    def _build_checkpoint_callback(self) -> ModelCheckpoint:
        checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=Path(self.config.paths.model_name).stem,
            monitor="valid/loss",
            mode="min",
            save_top_k=1,
        )

    def _build_lightning_module(self) -> ClassifierModule:
        return ClassifierModule(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            train_metric=self.train_metric,
            valid_metric=self.valid_metric,
            lr_scheduler_config=self.lr_scheduler_config,
        )

    @staticmethod
    def summarize_history(history: Dict[str, List[float]]) -> Dict[str, float]:
        best_epoch = min(
            range(len(history["valid_loss"])),
            key=lambda i: history["valid_loss"][i],
        )
        return {
            "epochs_ran": len(history["valid_loss"]),
            "best_epoch": best_epoch + 1,
            "best_valid_loss": float(history["valid_loss"][best_epoch]),
            "best_valid_acc": float(history["valid_acc"][best_epoch]),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_train_acc": float(history["train_acc"][-1]),
            "final_valid_loss": float(history["valid_loss"][-1]),
            "final_valid_acc": float(history["valid_acc"][-1]),
        }

    def run(
        self,
        version_name: str,
        experiments_xlsx: Optional[Path] = None,
        exp_name: Optional[str] = None,
    ) -> Dict[str, float]:
        matmul_precision = str(self.config.train.matmul_precision).strip().lower()
        if matmul_precision not in {"medium", "high"}:
            raise ValueError(
                "train.matmul_precision must be 'medium' or 'high'"
            )
        torch.set_float32_matmul_precision(matmul_precision)

        pl.seed_everything(self.config.train.seed, workers=True)
        accelerator, devices = resolve_accelerator_and_devices(
            self.config.train.device
        )

        trainer = pl.Trainer(
            max_epochs=self.config.train.epochs,
            accelerator=accelerator,
            devices=devices,
            logger=self.logger,
            callbacks=[self.checkpoint_callback, self.history_callback],
            deterministic=True,
            enable_checkpointing=True,
            log_every_n_steps=20,
        )
        trainer.fit(self.lightning_module, datamodule=self.data_module)

        if self.checkpoint_callback.best_model_path:
            output_path = export_best_state_dict(
                model=self.model,
                best_checkpoint_path=self.checkpoint_callback.best_model_path,
                path_config=self.config.paths,
            )
            print(f"Saved best checkpoint -> {output_path}")

        history = self.history_callback.history
        metrics_data = self.summarize_history(history)
        metrics_data["exp_name"] = exp_name or version_name

        target_experiments_xlsx = Path(
            experiments_xlsx or self.config.paths.experiments_file
        )
        record = append_experiment_record(
            excel_path=target_experiments_xlsx,
            config_data=self.config.to_dict(),
            metrics_data=metrics_data,
            version_name=version_name,
        )

        print("Experiment complete")
        print(f"Best valid loss approx: {min(history['valid_loss']):.4f}")
        print(f"Saved experiment record to: {target_experiments_xlsx}")
        print(f"version_name: {record['version_name']}")

        return {
            "exp_name": metrics_data["exp_name"],
            "version_name": record["version_name"],
            **metrics_data,
        }
