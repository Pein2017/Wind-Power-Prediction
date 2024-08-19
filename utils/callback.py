import datetime
import os

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from utils.config import dict_to_namespace
from utils.inference import full_inference


class SmoothedEarlyStopping(EarlyStopping):
    def __init__(self, *args, smoothing_factor=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing_factor = smoothing_factor
        self.smoothed_val_loss = None

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_val_loss = logs.get(self.monitor)

        if current_val_loss is None:
            return

        if self.smoothed_val_loss is None:
            self.smoothed_val_loss = current_val_loss
        else:
            self.smoothed_val_loss = (
                self.smoothing_factor * self.smoothed_val_loss
                + (1 - self.smoothing_factor) * current_val_loss
            )

        logs[self.monitor] = self.smoothed_val_loss

        super().on_validation_end(trainer, pl_module)


def get_callbacks(callback_settings):
    if isinstance(callback_settings, dict):
        callback_settings = dict_to_namespace(callback_settings)

    checkpoint_callback = ModelCheckpoint(
        monitor="Loss/val",
        filename="best_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=False,
    )

    early_stopping_callback = SmoothedEarlyStopping(
        monitor="Loss/val",
        patience=callback_settings.early_stop_patience,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=False,
        smoothing_factor=0.9,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    if checkpoint_callback:
        pass
    if lr_monitor:
        pass

    # lr_monitor
    return [
        early_stopping_callback,
        checkpoint_callback,
    ]


class MetricsCallback(Callback):
    def __init__(self, criterion, final_best_metrics_log_path):
        super().__init__()
        self.criterion = criterion

        self.log_path = final_best_metrics_log_path

    def on_validation_epoch_end(self, trainer, pl_module):
        self._update_metrics(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        self._output_best_metrics(pl_module, log_to_file=True)

    def _update_metrics(self, trainer, pl_module):
        current_phase = getattr(pl_module, "current_phase", None)

        if current_phase not in ["train", "val", "test"]:
            raise ValueError(
                "Invalid phase detected. Expected 'train', 'val', or 'test'."
            )

        current_train_loss = getattr(pl_module, "avg_train_loss", float("inf"))
        current_val_loss = getattr(pl_module, "avg_val_loss", float("inf"))
        current_test_loss = getattr(pl_module, "avg_test_loss", float("inf"))

        metric_update_flag = (
            current_train_loss < pl_module.curr_min_train_loss
            or current_val_loss < pl_module.curr_min_vali_loss
            or current_test_loss < pl_module.curr_min_test_loss
        )

        if metric_update_flag:
            train_loader = trainer.train_dataloader
            val_loader = trainer.val_dataloaders[0]
            test_loader = trainer.val_dataloaders[1]
            scaler = getattr(trainer, "scaler_y", None)
            scale_y_flag = True if scaler is not None else False
            (
                train_rmse,
                train_custom_acc,
                val_rmse,
                val_custom_acc,
                test_rmse,
                test_custom_acc,
            ) = full_inference(
                pl_module,
                train_loader,
                val_loader,
                test_loader,
                self.criterion,
                scaler,
                scale_y_flag,
            )

            # Update metrics
            self._update_best_metrics(
                pl_module,
                current_train_loss,
                current_val_loss,
                current_test_loss,
                train_rmse,
                train_custom_acc,
                val_rmse,
                val_custom_acc,
                test_rmse,
                test_custom_acc,
            )

    def _update_best_metrics(
        self,
        pl_module,
        current_train_loss,
        current_val_loss,
        current_test_loss,
        train_rmse,
        train_custom_acc,
        val_rmse,
        val_custom_acc,
        test_rmse,
        test_custom_acc,
    ):
        # Determine if there is any improvement
        train_improved = current_train_loss < pl_module.curr_min_train_loss
        val_improved = current_val_loss < pl_module.curr_min_vali_loss
        test_improved = current_test_loss < pl_module.curr_min_test_loss

        if train_improved:
            self._update_train_metrics(
                pl_module,
                current_train_loss,
                train_rmse,
                train_custom_acc,
                val_rmse,
                val_custom_acc,
                test_rmse,
                test_custom_acc,
            )

        if val_improved:
            self._update_val_metrics(
                pl_module,
                current_val_loss,
                train_rmse,
                train_custom_acc,
                val_rmse,
                val_custom_acc,
                test_rmse,
                test_custom_acc,
            )

        if test_improved:
            self._update_test_metrics(
                pl_module,
                current_test_loss,
                train_rmse,
                train_custom_acc,
                val_rmse,
                val_custom_acc,
                test_rmse,
                test_custom_acc,
            )

            # Print the updated metrics
            print(f"Epoch {pl_module.current_epoch + 1}:")
            if train_improved:
                print(f"train loss improved to {current_train_loss:.5f}")
            if val_improved:
                print(f"val loss improved to {current_val_loss:.5f}")
            if test_improved:
                print(f"test loss improved to {current_test_loss:.5f}")

            print(
                f"train loss: {pl_module.curr_min_train_loss:.5f} | val loss: {pl_module.curr_min_vali_loss:.5f} | test loss: {pl_module.curr_min_test_loss:.5f}"
            )
            self._output_best_metrics(pl_module, log_to_console=True)

    def _update_train_metrics(
        self,
        pl_module,
        current_train_loss,
        train_rmse,
        train_custom_acc,
        val_rmse,
        val_custom_acc,
        test_rmse,
        test_custom_acc,
    ):
        # Ensure the metric is improved before updating
        if train_rmse < pl_module.best_metrics.get("train_rmse", float("inf")):
            # scaled loss, without inverse transform
            pl_module.curr_min_train_loss = current_train_loss
            # unscaled loss, with inverse transform
            pl_module.best_metrics["train_rmse"] = train_rmse
            pl_module.best_metrics["train_custom_acc"] = train_custom_acc
            pl_module.best_metrics["val_rmse_for_best_train"] = val_rmse
            pl_module.best_metrics["val_custom_acc_for_best_train"] = val_custom_acc
            pl_module.best_metrics["test_rmse_for_best_train"] = test_rmse
            pl_module.best_metrics["test_custom_acc_for_best_train"] = test_custom_acc
            pl_module.best_metrics["train_epoch_for_best_train"] = (
                pl_module.current_epoch + 1
            )
        else:
            print(
                f"Warning: Train RMSE ({train_rmse}) is not less than the current best ({pl_module.best_metrics.get('train_rmse', float('inf'))})."
            )

    def _update_val_metrics(
        self,
        pl_module,
        current_val_loss,
        train_rmse,
        train_custom_acc,
        val_rmse,
        val_custom_acc,
        test_rmse,
        test_custom_acc,
    ):
        # Ensure the metric is improved before updating
        if val_rmse < pl_module.best_metrics.get("val_rmse", float("inf")):
            pl_module.curr_min_vali_loss = current_val_loss
            pl_module.best_metrics["val_rmse"] = val_rmse
            pl_module.best_metrics["val_custom_acc"] = val_custom_acc
            pl_module.best_metrics["train_rmse_for_best_val"] = train_rmse
            pl_module.best_metrics["train_custom_acc_for_best_val"] = train_custom_acc
            pl_module.best_metrics["test_rmse_for_best_val"] = test_rmse
            pl_module.best_metrics["test_custom_acc_for_best_val"] = test_custom_acc
            pl_module.best_metrics["val_epoch_for_best_val"] = (
                pl_module.current_epoch + 1
            )
        else:
            print(
                f"Warning: Val RMSE ({val_rmse}) is not less than the current best ({pl_module.best_metrics.get('val_rmse', float('inf'))})."
            )

    def _update_test_metrics(
        self,
        pl_module,
        current_test_loss,
        train_rmse,
        train_custom_acc,
        val_rmse,
        val_custom_acc,
        test_rmse,
        test_custom_acc,
    ):
        # Ensure the metric is improved before updating
        if test_rmse < pl_module.best_metrics.get("test_rmse", float("inf")):
            pl_module.curr_min_test_loss = current_test_loss
            pl_module.best_metrics["test_rmse"] = test_rmse
            pl_module.best_metrics["test_custom_acc"] = test_custom_acc
            pl_module.best_metrics["train_rmse_for_best_test"] = train_rmse
            pl_module.best_metrics["train_custom_acc_for_best_test"] = train_custom_acc
            pl_module.best_metrics["val_rmse_for_best_test"] = val_rmse
            pl_module.best_metrics["val_custom_acc_for_best_test"] = val_custom_acc
            pl_module.best_metrics["test_epoch_for_best_test"] = (
                pl_module.current_epoch + 1
            )
        else:
            print(
                f"Warning: Test RMSE ({test_rmse}) is not less than the current best ({pl_module.best_metrics.get('test_rmse', float('inf'))})."
            )

    def get_last_order_number(self):
        if not os.path.exists(self.log_path):
            return 0
        with open(self.log_path, "r") as f:
            lines = f.readlines()[-50:]  # Read only the last 50 lines
            for line in reversed(lines):
                if line.strip().startswith("Exp:"):
                    return int(line.split(":")[1].strip())
        return 0

    def _output_best_metrics(self, pl_module, log_to_console=False, log_to_file=False):
        exp_settings = pl_module.args.exp_settings
        best_metrics = pl_module.best_metrics

        metrics = [
            (
                "Train RMSE",
                best_metrics["train_rmse"],
                best_metrics["val_rmse_for_best_train"],
                best_metrics["test_rmse_for_best_train"],
                best_metrics["train_epoch_for_best_train"],
            ),
            (
                "Val RMSE",
                best_metrics["train_rmse_for_best_val"],
                best_metrics["val_rmse"],
                best_metrics["test_rmse_for_best_val"],
                best_metrics["val_epoch_for_best_val"],
            ),
            (
                "Test RMSE",
                best_metrics["train_rmse_for_best_test"],
                best_metrics["val_rmse_for_best_test"],
                best_metrics["test_rmse"],
                best_metrics["test_epoch_for_best_test"],
            ),
            (
                "Best Train Custom Accuracy",
                best_metrics["train_custom_acc"],
                best_metrics["val_custom_acc_for_best_train"],
                best_metrics["test_custom_acc_for_best_train"],
                best_metrics["train_epoch_for_best_train"],
            ),
            (
                "Best Val Custom Accuracy",
                best_metrics["train_custom_acc_for_best_val"],
                best_metrics["val_custom_acc"],
                best_metrics["test_custom_acc_for_best_val"],
                best_metrics["val_epoch_for_best_val"],
            ),
            (
                "Best Test Custom Accuracy",
                best_metrics["train_custom_acc_for_best_test"],
                best_metrics["val_custom_acc_for_best_test"],
                best_metrics["test_custom_acc"],
                best_metrics["test_epoch_for_best_test"],
            ),
        ]

        output = []
        output.append(f"\nExperiment Settings: {exp_settings}")
        if log_to_file:
            output.append(f"Experiment time: {datetime.datetime.now()}")
        output.append(
            f"{'+' + '-'*30 + '+' + '-'*15 + '+' + '-'*15 + '+' + '-'*15 + '+' + '-'*7 + '+'}"
        )
        output.append(
            f"| {'Metric':<28} | {'Training':<13} | {'Validation':<13} | {'Test':<13} | {'Epoch':<5} |"
        )
        output.append(
            f"{'+' + '-'*30 + '+' + '-'*15 + '+' + '-'*15 + '+' + '-'*15 + '+' + '-'*7 + '+'}"
        )

        for metric in metrics:
            metric_name, train_val, val_val, test_val, epoch = metric
            train_val_str = f"{train_val:.4f}" if train_val is not None else ""
            val_val_str = f"{val_val:.4f}" if val_val is not None else ""
            test_val_str = f"{test_val:.4f}" if test_val is not None else ""
            epoch_str = f"{epoch}" if epoch is not None else ""
            output.append(
                f"| {metric_name:<28} | {train_val_str:<13} | {val_val_str:<13} | {test_val_str:<13} | {epoch_str:<5} |"
            )

        output.append(
            f"{'+' + '-'*30 + '+' + '-'*15 + '+' + '-'*15 + '+' + '-'*15 + '+' + '-'*7 + '+'}"
        )

        if log_to_console:
            for line in output:
                print(line)

        if log_to_file:
            # Ensure the directory exists
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            last_order_number = self.get_last_order_number()
            order_number = last_order_number + 1
            with open(self.log_path, "a") as f:
                f.write(f"Exp: {order_number}\n")
                for line in output:
                    f.write(line + "\n")
                f.write("\n")


class OptunaPruningCallback(Callback):
    def __init__(self, trial):
        self.trial = trial

    def on_validation_end(self, trainer, pl_module):
        # Check if 'Loss/train' is in callback_metrics and is not None
        if (
            "Loss/train" not in trainer.callback_metrics
            or trainer.callback_metrics["Loss/train"] is None
        ):
            return  # Skip this epoch if 'Loss/train' is not set

        train_loss = trainer.callback_metrics["Loss/train"].item()
        val_loss = trainer.callback_metrics["Loss/val"].item()
        test_loss = trainer.callback_metrics["Loss/test"].item()

        weighted_loss = 0.01 * train_loss + 0.5 * val_loss + 0.5 * test_loss

        self.trial.report(weighted_loss, trainer.current_epoch)
        if self.trial.should_prune():
            print("Trial pruned!")
            trainer.should_stop = True
