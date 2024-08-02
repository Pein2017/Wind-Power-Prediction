# ruff: noqa: E402


from argparse import Namespace
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from models import SimpleMLP, TimeMixer
from utils.config import dict_to_namespace


class WindPowerExperiment(pl.LightningModule):
    def __init__(self, args: Namespace | Dict):
        super(WindPowerExperiment, self).__init__()
        if isinstance(args, dict):
            args = dict_to_namespace(args, False)
        self.args = args
        self.model_dict = {
            "TimeMixer": TimeMixer,
            "SimpleMLP": SimpleMLP,
        }

        if args:
            self.save_hyperparameters(args)
            # self.save_hyperparameters(vars(config))

        self.model = self._build_model()
        self.criterion = self._select_criterion()
        self.configure_metrics()

    def configure_metrics(self):
        # Initialize metrics
        self.curr_min_train_loss = float("inf")
        self.curr_min_vali_loss = float("inf")
        self.curr_min_test_loss = float("inf")

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.best_metrics = {
            # best train
            "train_rmse": float("inf"),
            "train_custom_acc": float("-inf"),
            "val_rmse_for_best_train": float("inf"),
            "val_custom_acc_for_best_train": float("-inf"),
            "test_rmse_for_best_train": float("inf"),
            "test_custom_acc_for_best_train": float("-inf"),
            "train_epoch_for_best_train": -1,
            # best val
            "val_rmse": float("inf"),
            "val_custom_acc": float("-inf"),
            "train_rmse_for_best_val": float("inf"),
            "train_custom_acc_for_best_val": float("-inf"),
            "test_rmse_for_best_val": float("inf"),
            "test_custom_acc_for_best_val": float("-inf"),
            "val_epoch_for_best_val": -1,
            # best test
            "test_rmse": float("inf"),
            "test_custom_acc": float("-inf"),
            "train_rmse_for_best_test": float("inf"),
            "train_custom_acc_for_best_test": float("-inf"),
            "val_rmse_for_best_test": float("inf"),
            "val_custom_acc_for_best_test": float("-inf"),
            "test_epoch_for_best_test": -1,
        }

    def common_step(self, batch, batch_idx, phase, dataloader_idx=None):
        loss = self.process_batch(batch, self.criterion)

        if phase == "train":
            self.train_losses.append(loss)

            # Log the learning rate
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]

            self.log("lr", current_lr, on_epoch=True, on_step=False, logger=True)
        elif phase == "val":
            self.val_losses.append(loss)
        elif phase == "test":
            self.test_losses.append(loss)

        # Set the current phase for metric callback
        self.current_phase = phase

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        phase = "val" if dataloader_idx == 0 else "test"
        return self.common_step(batch, batch_idx, phase, dataloader_idx)

    def on_validation_epoch_end(self):
        log_data = {}

        # Handling training loss logging
        if self.train_losses:
            avg_train_loss = self._aggregate_losses(self.train_losses)
            self.train_losses.clear()
            log_data["Loss/train"] = avg_train_loss
            self.avg_train_loss = avg_train_loss

        # Handling validation loss logging
        if self.val_losses:
            avg_val_loss = self._aggregate_losses(self.val_losses)
            self.val_losses.clear()
            log_data["Loss/val"] = avg_val_loss
            self.avg_val_loss = avg_val_loss

        # Handling test loss logging
        if self.test_losses:
            avg_test_loss = self._aggregate_losses(self.test_losses)
            self.test_losses.clear()
            log_data["Loss/test"] = avg_test_loss
            self.avg_test_loss = avg_test_loss

        # Log all at once
        self.log_dict(log_data, on_epoch=True, on_step=False)

    # def on_train_epoch_end(self):
    #     if not self.train_losses:
    #         raise ValueError("No training loss found for epoch end logging.")
    #     avg_train_loss = self._aggregate_losses(self.train_losses)
    #     self.avg_train_loss = avg_train_loss
    #     self.train_losses.clear()

    #     # Log to TensorBoard with truncated value
    #     truncated_train_loss = self._truncate_loss(avg_train_loss)
    #     print(f"debug: truncated_train_loss: {truncated_train_loss}")
    #     self.log(
    #         "train_loss",
    #         truncated_train_loss,
    #         on_epoch=True,
    #         on_step=False,
    #         logger=True,
    #     )

    #     # todo check if this is necessary
    #     # # Adjust the learning rate at the end of each epoch
    #     # scheduler = (
    #     #     self.lr_schedulers() if self.config.type == "OneCycleLR" else None
    #     # )
    #     # adjust_learning_rate(
    #     #     self.trainer.optimizers[0], scheduler, self.current_epoch, self.config
    #     # )

    def _truncate_loss(self, loss, max_value=5.0):
        return torch.clamp(loss, max=max_value)

    def _aggregate_losses(self, losses):
        # Handle both single tensor and list of tensors cases
        if isinstance(losses[0], list):
            losses = [item for sublist in losses for item in sublist]
        return torch.stack(losses).mean()

    def process_batch(self, batch, criterion, mask_value=None):
        batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
        dec_inp = None
        outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if mask_value is not None:
            mask = batch_y == mask_value
            outputs = torch.where(mask, torch.zeros_like(outputs), outputs)
        loss = self._compute_loss(outputs, batch_y, criterion)
        return loss

    def _prepare_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
        )

    def _compute_loss(self, outputs, batch_y, criterion):
        outputs = outputs[:, :, -1]
        batch_y = batch_y[:, :, -1].detach()
        mask = ~torch.isnan(batch_y)
        return criterion(outputs[mask], batch_y[mask])

    def _build_model(self):
        model_settings = self.args.model_settings
        model = self.model_dict[model_settings.name].Model(model_settings).float()
        return model

    def _select_optimizer(self):
        training_settings = self.args.training_settings
        scheduler_settings = self.args.scheduler_settings
        return optim.AdamW(
            self.model.parameters(),
            lr=training_settings.learning_rate,
            weight_decay=scheduler_settings.weight_decay,
        )

    def _select_criterion(self):
        return (
            nn.MSELoss() if self.args.training_settings.loss == "MSE" else nn.L1Loss()
        )

    def configure_optimizers(self):
        optimizer = self._select_optimizer()
        training_settings = self.args.training_settings
        scheduler_settings = self.args.scheduler_settings
        if scheduler_settings.type == "OneCycleLR":
            scheduler = {
                "scheduler": lr_scheduler.OneCycleLR(
                    optimizer=optimizer,
                    epochs=training_settings.train_epochs,
                    steps_per_epoch=training_settings.steps_per_epoch,
                    pct_start=scheduler_settings.pct_start,
                    max_lr=training_settings.learning_rate,
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.args.type == "CosineAnnealingLR":
            scheduler = {
                "scheduler": lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=scheduler_settings.T_max,
                    eta_min=int(scheduler_settings.eta_min),
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            scheduler = None

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
