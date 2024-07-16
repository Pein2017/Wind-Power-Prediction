# ruff: noqa: E402

import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

sys.path.append("/data3/lsf/Pein/Power-Prediction")


from dataset.datamodule import WindPowerDataModule
from exp.pl_exp import WindPowerExperiment
from utils.callback import MetricsCallback, get_callbacks
from utils.results import custom_test
from utils.tools import get_next_version


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def setup_device_and_strategy(args):
    """Setup device and strategy for training."""
    if args.use_multi_gpu:
        devices = torch.cuda.device_count()
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        devices = [int(args.gpu)]
        strategy = "auto"
    return devices, strategy


def execute_experiment(args):
    """Run a single experiment with the given configuration."""
    set_seed(args.seed)

    exp_output_dir = os.path.join(args.output_dir, args.exp_settings)
    os.makedirs(exp_output_dir, exist_ok=True)

    # Get the next version
    version = get_next_version(exp_output_dir)

    tb_logger = TensorBoardLogger(save_dir=exp_output_dir, name=None, version=version)

    devices, strategy = setup_device_and_strategy(args)

    data_module = WindPowerDataModule(args)
    data_module.prepare_data()
    data_module.setup("fit")

    # Calculate steps_per_epoch from train_dataloader
    train_dataloader = data_module.train_dataloader()
    args.steps_per_epoch = len(train_dataloader)

    # Create validation dataloaders list
    val_dataloaders = [data_module.val_dataloader(), data_module.test_dataloader()]

    # Initialize model
    wind_power_module = WindPowerExperiment(args)

    # Get callbacks
    callbacks = get_callbacks(args)
    callbacks.append(
        MetricsCallback(
            criterion=wind_power_module.criterion,
            final_best_metrics_log_path=args.final_best_metrics_log_path,
        )
    )

    # Determine sync_batchnorm_flag
    sync_batchnorm_flag = (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=[tb_logger],
        enable_progress_bar=False,
        sync_batchnorm=sync_batchnorm_flag,
        log_every_n_steps=args.batch_size,
        gradient_clip_val=getattr(args, "gradient_clip_val", None),
    )

    trainer.scaler_y = data_module.scaler_y

    # Train the model
    trainer.fit(
        wind_power_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloaders,
    )

    # Custom test the model and visualize results
    custom_test(
        trainer=trainer,
        model_class=WindPowerExperiment,
        data_module=data_module,
        exp_settings=args.exp_settings,
        device=torch.device("cpu"),
        best_metrics_dir=os.path.join(exp_output_dir, f"version_{version}"),
        plot_dir=os.path.join(exp_output_dir, f"version_{version}"),
    )
