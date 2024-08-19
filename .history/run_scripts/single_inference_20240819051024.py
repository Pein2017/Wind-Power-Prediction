# ruff: noqa: E402

import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

sys.path.append("/data/Pein/Pytorch/Wind-Power-Prediction")


import numpy as np
import pandas as pd

from dataset.datamodule import WindPowerDataModule
from exp.pl_exp import WindPowerExperiment
from utils.callback import MetricsCallback, get_callbacks
from utils.config import load_config
from utils.inference import inverse_transform
from utils.results import collect_preds, custom_test
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

    exp_output_dir = os.path.join(args.res_output_dir, args.exp_settings)
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
        device=torch.device("cuda"),
        best_metrics_dir=os.path.join(exp_output_dir, f"version_{version}"),
        plot_dir=os.path.join(exp_output_dir, f"version_{version}"),
        config=args,
    )


def load_model_from_checkpoint(config, checkpoint_path):
    """Load model from checkpoint."""
    model = WindPowerExperiment(config)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    return model


def save_results(test_data, preds, trues, output_csv_path):
    """Save results to a CSV file."""
    result_df = pd.DataFrame(
        {
            "time": test_data["time"],
            "True": trues,
            "Prediction": preds,
        }
    )
    result_df.to_csv(output_csv_path, index=False)
    print(f"Inference results saved to {output_csv_path}")


def find_checkpoint(checkpoints_dir):
    """Find the checkpoint file in the checkpoints directory."""
    for file_name in os.listdir(checkpoints_dir):
        if file_name.endswith(".ckpt"):
            return os.path.join(checkpoints_dir, file_name)
    raise FileNotFoundError("No checkpoint file found in the directory.")


def main_inference(exp_result_dir, res_output_dir):
    """Main function to conduct inference and save predictions to a CSV file."""
    # Paths
    config_path = os.path.join(exp_result_dir, "hparams.yaml")
    checkpoint_path = find_checkpoint(os.path.join(exp_result_dir, "checkpoints"))

    if not os.path.exists(res_output_dir):
        os.makedirs(res_output_dir)

    output_csv_path = os.path.join(res_output_dir, "inference_results.csv")

    # Load the configuration from the YAML file
    config = load_config(config_path)

    # Set seed for reproducibility
    set_seed(config["seed"])

    test_data_path = os.path.join(config["data_root_dir"], config["test_path"])
    test_data = pd.read_csv(test_data_path)
    # remove first config['seq_len'] rows of test_data to match the length of the test_dataloader
    test_data = test_data.iloc[config["seq_len"] :]

    # Prepare data module
    data_module = WindPowerDataModule(config)
    data_module.prepare_data()
    data_module.setup("test")

    # Ensure data length matches
    expected_length = len(data_module.test_dataloader().dataset)
    if len(test_data) != expected_length:
        print(f"Expected length based on dataloader dataset: {expected_length}")
        print(f"But actual length of test data: {len(test_data)}")
        raise AssertionError("Length mismatch between test data and dataloader.")

    # Load model from checkpoint
    model = load_model_from_checkpoint(config, checkpoint_path)

    # Run inference
    preds, trues = collect_preds(model, data_module.test_dataloader())

    # Set all the negative values of preds to 0
    preds[preds < 0] = 0

    # Check if predictions match the true values after inverse transformation
    if data_module.scaler_y:
        inversed_preds, inversed_trues = inverse_transform(
            preds, trues, data_module.scaler_y
        )
        assert np.allclose(inversed_trues, test_data["power"].values, atol=1e-5), (
            "Mismatch between inverse transformed true values and test data power column. "
            f"Max difference: {np.max(np.abs(inversed_trues - test_data['power'].values))}"
        )
    else:
        inversed_preds, inversed_trues = preds, trues
        assert np.allclose(trues, test_data["power"].values, atol=1e-5), (
            "Mismatch between true values and test data power column. "
            f"Max difference: {np.max(np.abs(trues - test_data['power'].values))}"
        )

    print(f"debug: output_csv_path is {output_csv_path}")
    # Save results
    save_results(test_data, inversed_preds, inversed_trues, output_csv_path)

    # rmse
    print(f"\n\nRMSE: {np.sqrt(np.mean((inversed_preds - inversed_trues) ** 2))}")


if __name__ == "__main__":
    time_str = "24-07-17"
    exp_result_dir = f"/data/Pein/Pytorch/Wind-Power-Prediction/res_output/{time_str}/seq_len-36-lr-0.1-d-256-hid_d-32-last_d-256-time_d-128-e_layers-8-comb_type-add-bs-1100/version_0/"
    res_output_dir = (
        f"/data/Pein/Pytorch/Wind-Power-Prediction/res_output/{time_str}_result/best_preds/"
    )

    main_inference(exp_result_dir, res_output_dir)
