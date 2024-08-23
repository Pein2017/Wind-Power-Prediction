# ruff: noqa E402
import contextlib
import json
import multiprocessing  # noqa
import os
import sqlite3
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict

import optuna
import torch
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import CmaEsSampler, GridSampler, NSGAIISampler, RandomSampler

# from optuna.integration import PyTorchLightningPruningCallback

sys.path.append("/data/Pein/Pytorch/Wind-Power-Prediction")

import pytorch_lightning as pl  # noqa

from exp.pl_exp import WindPowerExperiment
from run_scripts.single_exp import (
    prepare_data_module,
    prepare_output_directory,
    set_seed,
    setup_device_and_strategy,
    setup_logging,
    setup_trainer,
)
from utils.callback import MetricsCallback, OptunaPruningCallback, get_callbacks
from utils.config import dict_to_namespace, load_config, namespace_to_dict
from utils.results import custom_test
from utils.tools import get_next_version


def setup_environment(trial, args):
    """Set up the environment for the experiment based on the trial and configuration."""
    set_seed(args.general_settings.seed)
    exp_output_dir = prepare_output_directory(args)
    version = get_next_version(exp_output_dir)

    use_wandb = getattr(args.logging_settings, "use_wandb", False)
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
    callbacks.append(OptunaPruningCallback(trial))

    trainer = setup_trainer(args, devices, strategy, logger, callbacks)
    trainer.scaler_y = data_module.scaler_y

    return (
        trainer,
        wind_power_module,
        train_dataloader,
        val_dataloaders,
        use_wandb,
        exp_output_dir,
        version,
        data_module,
        args,
    )


# Define the mapping of hyperparameters to the nested paths
hyperparam_path_map = {
    # Model settings
    "model_settings": {
        "d_model": "d_model",
        "hidden_d_model": "hidden_d_model",
        "token_conv_kernel": "token_conv_kernel",
        "last_d_model": "last_d_model",
        "time_d_model": "time_d_model",
        "pos_d_model": "pos_d_model",
        "seq_len": "seq_len",
        "e_layers": "e_layers",
        "seq_layers": "seq_layers",
        "dropout": "dropout",
        "combine_type": "combine_type",
        "use_pos_enc": "use_pos_enc",
        "bidirectional": "bidirectional",
        "norm_type": "norm_type",
        "num_heads": "num_heads",
        "fc_layer_type": "fc_layer_type",
        "token_d_model": "token_d_model",
        "conv_out_dim": "conv_out_dim",
        "feat_conv_kernel": "feat_conv_kernel",
        "norm_after_dict": "norm_after_dict",
        "skip_connection_mode": "skip_connedsction_mode",
        "conv_norm": "conv_norm",
        "mlp_norm": "mlp_norm",
    },
    # Training settings
    "training_settings": {
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "train_epochs": "train_epochs",
    },
    "data_settings": {
        "scale_y_type": "scale_y_type",
        "random_split": "random_split",
        "val_split": "val_split",
        "y_transform_order": "y_transform_order",
    },
    "scheduler_settings": {
        "weight_decay": "weight_decay",
    },
}


def update_nested_dict(base_dict: Dict, updates, path_map):
    """Update nested dictionary based on the path map."""
    # Convert base_dict's nested Namespace objects to dictionaries
    base_dict = namespace_to_dict(base_dict)

    for section, params in path_map.items():
        for key, sub_key in params.items():
            if key in updates:
                if section in base_dict:
                    base_dict[section][sub_key] = updates[key]
                else:
                    raise KeyError(
                        f"Section '{section}' not found in base configuration."
                    )

    # Convert base_dict back to Namespace
    return dict_to_namespace(base_dict, True)


def _load_and_update_config(
    config_path: Path, trial: optuna.trial.Trial, time_str: str
):
    """Load configuration and update with suggested hyperparameters."""
    # Load the base configuration from YAML and replace time placeholders
    base_config = load_config(config_path=config_path, time_str=time_str)

    # Convert the base_config Namespace to a dictionary for updating
    base_config_dict = namespace_to_dict(base_config)

    # Get suggested hyperparameters from the trial
    suggested_hyperparams = suggest_hyperparameters(trial)

    # Debug: Print keys
    print("Base config keys:", base_config_dict.keys())
    print("Suggested hyperparameters keys:", suggested_hyperparams.keys())

    # Check and adjust hidden_d_model to be divisible by num_heads
    hidden_d_model = suggested_hyperparams.get("hidden_d_model")
    num_heads = suggested_hyperparams.get("num_heads")
    suggested_hyperparams["dropout"] = round(suggested_hyperparams["dropout"], 3)

    if hidden_d_model is not None and num_heads is not None:
        if hidden_d_model % num_heads != 0:
            # Reduce num_heads so that hidden_d_model is divisible by num_heads
            max_divisible_heads = max(
                [n for n in range(1, hidden_d_model + 1) if hidden_d_model % n == 0]
            )
            print(
                f"Adjusting num_heads from {num_heads} to {max_divisible_heads} "
                f"to ensure hidden_d_model {hidden_d_model} is divisible by num_heads."
            )
            suggested_hyperparams["num_heads"] = max_divisible_heads

    # Update the base_config dictionary with the suggested hyperparameters
    updated_config_dict = update_nested_dict(
        base_config_dict, suggested_hyperparams, hyperparam_path_map
    )

    # Convert the updated dictionary back to Namespace
    updated_config = dict_to_namespace(updated_config_dict, True)

    return updated_config


def _prepare_logging(base_config):
    """Prepare log file path and ensure directory exists."""
    log_file_path = (
        Path(base_config.output_paths.train_log_dir) / f"{base_config.exp_settings}.log"
    )
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    return log_file_path


def run_training(trainer, wind_power_module, train_dataloader, val_dataloaders):
    """Run the training process and return the duration."""
    start_time = time.time()
    trainer.fit(
        wind_power_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloaders,
    )
    end_time = time.time()
    training_duration = end_time - start_time
    training_duration_str = time.strftime("%H:%M:%S", time.gmtime(training_duration))
    print(f"Training time: {training_duration_str}")
    return training_duration


def run_custom_test(
    trainer, wind_power_module, data_module, base_config, exp_output_dir, version
):
    """Run the custom test and return the duration."""
    start_time = time.time()
    custom_test(
        trainer=trainer,
        model_class=WindPowerExperiment,
        data_module=data_module,
        exp_settings=base_config.exp_settings,
        device=torch.device("cuda"),
        best_metrics_dir=os.path.join(exp_output_dir, f"version_{version}"),
        plot_dir=os.path.join(exp_output_dir, f"version_{version}"),
        config=base_config,
    )
    end_time = time.time()
    testing_duration = end_time - start_time
    testing_duration_str = time.strftime("%H:%M:%S", time.gmtime(testing_duration))
    print(f"Custom testing time: {testing_duration_str}")
    return testing_duration


def calculate_total_duration(start_time):
    """Calculate and print the total duration of the trial."""
    end_time = time.time()
    total_duration = end_time - start_time
    total_duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))
    print(f"Total trial time: {total_duration_str}")
    return total_duration


def _log_metrics(trainer, use_wandb):
    """Log the metrics and integrate with WandB if enabled."""
    train_loss = trainer.callback_metrics["Loss/train"].item()
    val_loss = trainer.callback_metrics["Loss/val"].item()
    test_loss = trainer.callback_metrics["Loss/test"].item()

    weighted_loss = 0.01 * train_loss + 0.5 * val_loss + 0.5 * test_loss

    if use_wandb:
        import wandb

        wandb.log({"Loss/val": val_loss})
        wandb.run.finish()

    return weighted_loss


def objective(trial, config_path: str, time_str):
    """Objective function for Optuna to minimize."""
    # try:
    base_config = _load_and_update_config(config_path, trial, time_str)
    print(f"\nRunning experiment with config:\n{base_config}\n\n")

    (
        trainer,
        wind_power_module,
        train_dataloader,
        val_dataloaders,
        use_wandb,
        exp_output_dir,
        version,
        data_module,
        base_config,
    ) = setup_environment(trial, base_config)

    log_file_path = _prepare_logging(base_config)

    with open(log_file_path, "w") as f, contextlib.redirect_stdout(
        f
    ), contextlib.redirect_stderr(f):
        trial_start_time = time.time()

        training_duration = run_training(
            trainer, wind_power_module, train_dataloader, val_dataloaders
        )
        testing_duration = run_custom_test(
            trainer,
            wind_power_module,
            data_module,
            base_config,
            exp_output_dir,
            version,
        )

        calculate_total_duration(trial_start_time)

        print(f"Training time of trial: {training_duration:.2f} seconds")
        print(f"Testing time of trial: {testing_duration:.2f} seconds")
        print(
            f"Total time of trial: {training_duration + testing_duration:.2f} seconds"
        )

    weighted_loss = _log_metrics(trainer, use_wandb)

    return weighted_loss

    # except Exception:
    #     print("Trial failed due to an error:")
    #     traceback.print_exc()  # Print the full traceback of the error
    #     # Return a very high loss value to signify failure in Optuna
    #     return float("inf")


study_lock = Lock()


def retry_on_lock(func):
    def wrapper(*args, **kwargs):
        retries = 5
        delay = 1
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        raise sqlite3.OperationalError("Database is locked after multiple retries.")

    return wrapper


@retry_on_lock
def create_study(args):
    """Create an Optuna study with the specified configuration."""
    # Modify the connection string to use WAL mode
    storage_name = (
        f"sqlite:///{os.path.join(args.output_dir, f'{args.study_name}.db')}?mode=wal"
    )

    print(f'Creating study "{args.study_name}" with storage "{storage_name}"...')

    sampler = get_sampler(args)
    pruner = get_pruner(args)
    print(f"Using sampler : {sampler}, pruner : {pruner}")

    with study_lock:
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=storage_name,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

        # Explicitly enable WAL mode
        with sqlite3.connect(
            os.path.join(args.output_dir, f"{args.study_name}.db")
        ) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.commit()

        return study


def get_pruner(args):
    """Get the appropriate Optuna pruner based on the provided type."""
    if args.pruner_type == "median":
        return MedianPruner(n_startup_trials=10, n_warmup_steps=25, interval_steps=1)
    elif args.pruner_type == "hyperband":
        return HyperbandPruner(min_resource=3, max_resource="auto", reduction_factor=3)
    elif args.pruner_type == "successive_halving":
        return SuccessiveHalvingPruner(min_resource=3, reduction_factor=2)
    else:
        return optuna.pruners.NopPruner()


def get_sampler(args):
    """Get the appropriate Optuna sampler based on the provided name."""
    search_space = suggest_hyperparameters(return_search_space=True)
    seed = args.seed
    sampler_name = args.sampler_name
    print(f"Using sampler {sampler_name} with seed {seed}")
    if sampler_name == "cma":
        return CmaEsSampler(seed=seed)
    elif sampler_name == "random":
        return RandomSampler(seed=seed)
    elif sampler_name == "grid":
        return GridSampler(search_space=search_space, seed=seed)
    elif sampler_name == "nsgaii":
        return NSGAIISampler(seed=seed)
    else:
        return None


def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def print_best_trial(study):
    """Print the best trial's results."""
    best_trial = study.best_trial

    print("Best trial:")
    print(f"Value: {best_trial.value}")
    print("Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


def save_top_trials(study, output_dir, study_name):
    """Save the top N trials to a JSON file."""
    # Filter out trials with None as their value
    valid_trials = [t for t in study.trials if t.value is not None]

    # Determine the number of top trials to save
    top_N = min(10, len(valid_trials))

    # Get the top N trials
    top_trials = sorted(valid_trials, key=lambda t: t.value)[:top_N]
    top_hyperparams = [{t.number: t.params, "value": t.value} for t in top_trials]

    # Save the top N trials to a JSON file
    best_hyperparams_filepath = os.path.join(
        output_dir, f"{study_name}_top{top_N}_params.json"
    )

    with open(best_hyperparams_filepath, "w") as f:
        json.dump(top_hyperparams, f)

    print(f"Top {top_N} trials saved to {best_hyperparams_filepath}")


def run_optimization(study, config_path, n_trials, timeout, time_str):
    """Run the optimization process."""
    study.optimize(
        lambda trial: objective(trial, config_path, time_str=time_str),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )


def run_optuna_study(args):
    """Run Optuna study for hyperparameter optimization."""
    args = dict_to_namespace(args, if_create_exp_settings=False)
    ensure_output_dir(args.output_dir)

    study = create_study(args)
    run_optimization(
        study, args.config_path, args.n_trials, args.timeout, args.time_str
    )

    print(f"Number of finished trials: {len(study.trials)}")
    print_best_trial(study)
    save_top_trials(study, args.output_dir, args.study_name)


def main():
    import time

    import numpy as np  # noqa

    # Generate a time-based seed
    time_seed = int(time.time() * 10000) % 100000

    time_str = "24-08-23-test"
    study_name = f"{time_str}-farm_66"
    n_trails = 20 * 1
    sampler_name = "cma"
    pruner_type = "median"
    args = {
        "time_str": time_str,
        "study_name": study_name,
        "timeout": None,  # * keep this argument
        "n_trials": int(n_trails),
        "output_dir": f"/data/Pein/Pytorch/Wind-Power-Prediction/optuna_results/{time_str}",
        "config_path": "/data/Pein/Pytorch/Wind-Power-Prediction/config/optuna_config.yaml",
        "sampler_name": sampler_name,
        "seed": time_seed,
        "pruner_type": pruner_type,
    }
    run_optuna_study(args)


def suggest_hyperparameters(
    trial: optuna.trial.Trial = None, return_search_space=False
):
    import numpy as np  # noqa

    """Suggest hyperparameters using Optuna trial or return search space."""
    search_space = {
        # d_model related parameters
        "d_model": [64],
        "hidden_d_model": [64],
        "last_d_model": [64, 128],
        "time_d_model": [16],
        "pos_d_model": [32],
        "token_d_model": [8],
        # Model architecture and layers
        "e_layers": [3, 6],
        "token_conv_kernel": [7],
        "feat_conv_kernel": [11],
        "conv_out_dim": [64, 128],
        # Attention mechanism parameters
        "num_heads": [4, 8],
        # Miscellaneous fixed parameters
        "combine_type": ["add"],
        "use_pos_enc": [True, False],
        "norm_type": ["layer"],  # batch, layer
        "dropout": [0.1],
        "seq_len": [8],
        "train_epochs": [60],
        # Parameters to search
        "learning_rate": [8e-4, 8e-3, 2e-2],  # 8e-4, 1e-3,
        "batch_size": [1024],
        "conv_norm": [True],
        "mlp_norm": [True, False],
        "skip_connection_mode": [
            "conv_mlp",
        ],  # "none", "conv_mha", "conv_mlp", "full"
        "scale_y_type": ["standard"],  # standard, min_max
        "weight_decay": [1e-4],
        "val_split": [0.2],
        "y_transform_order": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    }

    if return_search_space:
        return search_space

    hyperparams = {
        # Integer suggestions
        "num_heads": trial.suggest_int(
            "num_heads",
            min(search_space["num_heads"]),
            max(search_space["num_heads"]),
            step=8,
        ),
        "hidden_d_model": trial.suggest_int(
            "hidden_d_model",
            min(search_space["hidden_d_model"]),
            max(search_space["hidden_d_model"]),
            step=4,
        ),
        "token_conv_kernel": trial.suggest_int(
            "token_conv_kernel",
            min(search_space["token_conv_kernel"]),
            max(search_space["token_conv_kernel"]),
            step=2,
        ),
        "last_d_model": trial.suggest_int(
            "last_d_model",
            min(search_space["last_d_model"]),
            max(search_space["last_d_model"]),
            step=64,
        ),
        "seq_len": trial.suggest_int(
            "seq_len",
            min(search_space["seq_len"]),
            max(search_space["seq_len"]),
        ),
        "token_d_model": trial.suggest_int(
            "token_d_model",
            min(search_space["token_d_model"]),
            max(search_space["token_d_model"]),
            step=4,
        ),
        "time_d_model": trial.suggest_int(
            "time_d_model",
            min(search_space["time_d_model"]),
            max(search_space["time_d_model"]),
            step=4,
        ),
        "pos_d_model": trial.suggest_int(
            "pos_d_model",
            min(search_space["pos_d_model"]),
            max(search_space["pos_d_model"]),
            step=4,
        ),
        "d_model": trial.suggest_int(
            "d_model",
            min(search_space["d_model"]),
            max(search_space["d_model"]),
            step=32,
        ),
        "conv_out_dim": trial.suggest_int(
            "conv_out_dim",
            min(search_space["conv_out_dim"]),
            max(search_space["conv_out_dim"]),
            step=64,
        ),
        "e_layers": trial.suggest_int(
            "e_layers",
            min(search_space["e_layers"]),
            max(search_space["e_layers"]),
            step=1,
        ),
        # Float suggestions
        "learning_rate": trial.suggest_float(
            "learning_rate",
            min(search_space["learning_rate"]),
            max(search_space["learning_rate"]),
            log=True,
        ),
        "dropout": trial.suggest_float(
            "dropout",
            min(search_space["dropout"]),
            max(search_space["dropout"]),
            step=0.02,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            min(search_space["weight_decay"]),
            max(search_space["weight_decay"]),
            log=True,
        ),
        # Categorical suggestions
        "train_epochs": trial.suggest_categorical(
            "train_epochs", search_space["train_epochs"]
        ),
        "combine_type": trial.suggest_categorical(
            "combine_type", search_space["combine_type"]
        ),
        "use_pos_enc": trial.suggest_categorical(
            "use_pos_enc", search_space["use_pos_enc"]
        ),
        "norm_type": trial.suggest_categorical("norm_type", search_space["norm_type"]),
        "batch_size": trial.suggest_categorical(
            "batch_size", search_space["batch_size"]
        ),
        "feat_conv_kernel": trial.suggest_categorical(
            "feat_conv_kernel", search_space["feat_conv_kernel"]
        ),
        "skip_connection_mode": trial.suggest_categorical(
            "skip_connection_mode", search_space["skip_connection_mode"]
        ),
        "conv_norm": trial.suggest_categorical("conv_norm", search_space["conv_norm"]),
        "mlp_norm": trial.suggest_categorical("mlp_norm", search_space["mlp_norm"]),
        "scale_y_type": trial.suggest_categorical(
            "scale_y_type", search_space["scale_y_type"]
        ),
        "val_split": trial.suggest_categorical("val_split", search_space["val_split"]),
        "y_transform_order": trial.suggest_categorical(
            "y_transform_order", search_space["y_transform_order"]
        ),
    }

    return hyperparams


if __name__ == "__main__":
    import os
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    print(
        f"Total time taken: {end_time - start_time: 0.4f} seconds, {(end_time - start_time) / 60: 0.2f} minutes"
    )
    print("Done!")
