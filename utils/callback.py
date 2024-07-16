import datetime

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from utils.inference import full_inference


def get_callbacks(config):
    checkpoint_callback = ModelCheckpoint(
        monitor="Loss/val",
        filename="best_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="Loss/val",
        patience=config.early_stop_patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    if checkpoint_callback:
        pass
    if lr_monitor:
        pass

    return [early_stopping_callback, checkpoint_callback, lr_monitor]


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
        improved_metrics = []

        if current_train_loss < pl_module.curr_min_train_loss:
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
            improved_metrics.append(f"train loss improved to {current_train_loss:.5f}")

        if current_val_loss < pl_module.curr_min_vali_loss:
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
            improved_metrics.append(f"val loss improved to {current_val_loss:.5f}")

        if current_test_loss < pl_module.curr_min_test_loss:
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
            improved_metrics.append(f"test loss improved to {current_test_loss:.5f}")

        if improved_metrics:
            print(f"Epoch {pl_module.current_epoch + 1}:")
            for metric in improved_metrics:
                print(metric)
            print(
                f"train loss: {pl_module.curr_min_train_loss:.5f} | val loss: {pl_module.curr_min_vali_loss:.5f} | test loss: {pl_module.curr_min_test_loss:.5f}"
            )
            self._output_best_metrics(pl_module, log_to_console=True)

    def _output_best_metrics(self, pl_module, log_to_console=False, log_to_file=False):
        exp_settings = pl_module.config.exp_settings
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
                None,
                None,
                best_metrics["train_epoch_for_best_train"],
            ),
            (
                "Best Val Custom Accuracy",
                None,
                best_metrics["val_custom_acc"],
                None,
                best_metrics["val_epoch_for_best_val"],
            ),
            (
                "Best Test Custom Accuracy",
                None,
                None,
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
            with open(self.log_path, "a") as f:
                for line in output:
                    f.write(line + "\n")
                f.write(line + "\n")
                f.write("\n")

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
        pl_module.curr_min_train_loss = current_train_loss
        pl_module.best_metrics["train_rmse"] = train_rmse
        pl_module.best_metrics["train_custom_acc"] = train_custom_acc
        pl_module.best_metrics["val_rmse_for_best_train"] = val_rmse
        # pl_module.best_metrics["val_custom_acc_for_best_train"] = val_custom_acc
        pl_module.best_metrics["test_rmse_for_best_train"] = test_rmse
        # pl_module.best_metrics["test_custom_acc_for_best_train"] = test_custom_acc
        pl_module.best_metrics["train_epoch_for_best_train"] = (
            pl_module.current_epoch + 1
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
        pl_module.curr_min_vali_loss = current_val_loss
        pl_module.best_metrics["val_rmse"] = val_rmse
        pl_module.best_metrics["val_custom_acc"] = val_custom_acc
        pl_module.best_metrics["train_rmse_for_best_val"] = train_rmse
        # pl_module.best_metrics["train_custom_acc_for_best_val"] = train_custom_acc
        pl_module.best_metrics["test_rmse_for_best_val"] = test_rmse
        # pl_module.best_metrics["test_custom_acc_for_best_val"] = test_custom_acc
        pl_module.best_metrics["val_epoch_for_best_val"] = pl_module.current_epoch + 1

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
        pl_module.curr_min_test_loss = current_test_loss
        pl_module.best_metrics["test_rmse"] = test_rmse
        pl_module.best_metrics["test_custom_acc"] = test_custom_acc
        pl_module.best_metrics["train_rmse_for_best_test"] = train_rmse
        # pl_module.best_metrics["train_custom_acc_for_best_test"] = train_custom_acc
        pl_module.best_metrics["val_rmse_for_best_test"] = val_rmse
        # pl_module.best_metrics["val_custom_acc_for_best_test"] = val_custom_acc
        pl_module.best_metrics["test_epoch_for_best_test"] = pl_module.current_epoch + 1
