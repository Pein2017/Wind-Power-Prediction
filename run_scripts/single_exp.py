# ruff: noqa: E402

import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

sys.path.append("/data3/lsf/Pein/Power-Prediction")

from dataset.datamodule import WindPowerDataModule
from exp.pl_exp import WindPowerExperiment
from utils.callback import MetricsCallback, get_callbacks
from utils.config import dict_to_namespace
from utils.results import custom_test
from utils.tools import get_next_version


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.enabled = False
    pl.seed_everything(seed, workers=True)


def setup_device_and_strategy(gpu_settings):
    """Setup device and strategy for training."""
    if isinstance(gpu_settings, dict):
        gpu_settings = dict_to_namespace(gpu_settings, False)
        print("Warning: args is still a dict, converting it to namespace")
    if gpu_settings.use_multi_gpu:
        devices = torch.cuda.device_count()
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        devices = [int(gpu_settings.gpu_id)]
        strategy = "auto"
    return devices, strategy


def prepare_output_directory(args):
    """Prepare the output directory for the experiment."""
    if isinstance(args, dict):
        args = dict_to_namespace(args, False)

    if args.exp_settings is None:
        raise ValueError("exp_settings is not specified in the config file.")

    exp_output_dir = os.path.join(args.output_paths.res_output_dir, args.exp_settings)
    os.makedirs(exp_output_dir, exist_ok=True)
    return exp_output_dir


def setup_logging(exp_output_dir, version=None, use_wandb=False):
    """Setup logger for the experiment."""
    if version is None:
        version = get_next_version(exp_output_dir)

    if use_wandb:
        os.environ["WANDB_DISABLED"] = "false"
        logger = WandbLogger(
            project="wind_power_prediction",
            save_dir=exp_output_dir,
            log_model=False,
        )
    else:
        logger = TensorBoardLogger(save_dir=exp_output_dir, name=None, version=version)
    return logger, version


def prepare_data_module(args):
    """Prepare the data module."""
    data_module = WindPowerDataModule(args)
    data_module.prepare_data()
    data_module.setup("fit")

    return data_module


def setup_trainer(args, devices, strategy, logger, callbacks):
    """Setup PyTorch Lightning Trainer."""
    if isinstance(args, dict):
        args = dict_to_namespace(args, False)

    # Check if the specified GPU (global ID 1, local ID 0) is available
    visible_devices = torch.cuda.device_count()
    if visible_devices < 1:
        raise RuntimeError(
            "No GPUs are available. Please check your CUDA_VISIBLE_DEVICES setting."
        )

    # Check if the intended GPU ID is within the available range
    intended_gpu_id = 0  # Local ID when CUDA_VISIBLE_DEVICES=1
    if intended_gpu_id >= visible_devices:
        raise RuntimeError(
            f"GPU with ID {intended_gpu_id} (global ID 1) is not available. Please check your CUDA_VISIBLE_DEVICES setting."
        )

    sync_batchnorm_flag = (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    )

    trainer = pl.Trainer(
        max_epochs=args.training_settings.train_epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        sync_batchnorm=sync_batchnorm_flag,
        gradient_clip_val=args.training_settings.gradient_clip_val,
    )
    return trainer


def execute_experiment(args):
    """Run a single experiment with the given configuration."""
    set_seed(args.general.seed)

    exp_output_dir = prepare_output_directory(args)
    version = get_next_version(exp_output_dir)

    use_wandb = args.logging.use_wandb  # Accessing use_wandb from logging settings
    logger, version = setup_logging(exp_output_dir, version, use_wandb)

    devices, strategy = setup_device_and_strategy(args.gpu_settings)

    data_module = prepare_data_module(args)

    train_dataloader = data_module.train_dataloader()
    args.training_settings.steps_per_epoch = len(train_dataloader)
    val_dataloaders = [data_module.val_dataloader(), data_module.test_dataloader()]

    wind_power_module = WindPowerExperiment(args)
    callbacks = get_callbacks(args.training_settings)
    callbacks.append(
        MetricsCallback(
            criterion=wind_power_module.criterion,
            final_best_metrics_log_path=args.output_paths.final_best_metrics_log_path,
        )
    )

    trainer = setup_trainer(args, devices, strategy, logger, callbacks)
    trainer.scaler_y = data_module.scaler_y

    trainer.fit(
        model=wind_power_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloaders,
    )

    custom_test(
        trainer=trainer,
        model_class=WindPowerExperiment,
        data_module=data_module,
        exp_settings=args.exp_settings,
        device=torch.device("cuda" if args.gpu_settings.use_gpu else "cpu"),
        best_metrics_dir=os.path.join(exp_output_dir, f"version_{version}"),
        plot_dir=os.path.join(exp_output_dir, f"version_{version}"),
        config=args,
    )
