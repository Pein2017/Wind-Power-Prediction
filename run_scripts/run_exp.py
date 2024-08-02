import os
import sys
import time
import traceback
from datetime import timedelta

import numpy as np  # noqa
import pathos.multiprocessing as mp
import torch

sys.path.append("/data3/lsf/Pein/Power-Prediction")


from run_scripts.run_optuna import suggest_hyperparameters
from run_scripts.single_exp import execute_experiment
from utils.config import dict_to_namespace, load_config, parse_args
from utils.exp_search import generate_hyperparameter_combinations

# Use the suggest_hyperparameters function for defining the search space
SEARCH_SPACE = suggest_hyperparameters(return_search_space=True)


def redirect_stdout_stderr(log_filepath):
    """Redirect stdout and stderr to a log file."""
    log_file = open(log_filepath, "w")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


def execute_experiment_with_logging(config):
    """Execute experiment with logging redirection."""

    # Build the log filename
    output_paths = config.output_paths
    model_settings = config.model_settings
    training_settings = config.training_settings

    log_filename = (
        f"seq_len-{model_settings.seq_len}-"
        f"lr-{training_settings.learning_rate}-"
        f"d-{model_settings.d_model}-"
        f"last_d-{model_settings.last_hidden_dim}-"
        f"time_d-{model_settings.time_d_model}"
    )

    if config.gpu_settings.use_multi_gpu:
        log_filename += "-allgpu.log"
    else:
        log_filename += f"-gpu-{config.gpu_settings.gpu_id}.log"

    os.makedirs(output_paths.train_log_dir, exist_ok=True)

    log_filepath = os.path.join(output_paths.train_log_dir, log_filename)

    with redirect_stdout_stderr(log_filepath) as log_file:  # noqa
        print(f"Running experiment with config: {config}\n")
        try:
            execute_experiment(config)
        except Exception as e:
            print(f"Error during experiment: {e}")
            traceback.print_exc(file=sys.stderr)
        finally:
            print("Experiment finished.")
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


def worker(config, device_id=None):
    """Worker function to run an experiment."""
    if device_id is not None:
        config.gpu_settings.gpu_id = device_id
    start_time = time.time()
    execute_experiment_with_logging(config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time


def run_experiments_in_order(configurations):
    total_experiments = len(configurations)
    for i, config in enumerate(configurations):
        print(f"Running {i+1}/{total_experiments} of the experiments.")
        try:
            exp_start_time = time.time()
            _ = worker(config)
            exp_end_time = time.time()
            total_elapsed_time = exp_end_time - exp_start_time
            print(
                f"Completed {i+1}/{total_experiments} of the experiments in {timedelta(seconds=total_elapsed_time)}."
            )
        except Exception as e:
            print(f"Error during experiment {i+1}: {e}")


def run_experiments_in_parallel(configurations, num_devices):
    start_time = time.time()
    with mp.Pool(processes=num_devices) as pool:
        try:
            _ = pool.starmap(
                worker,
                [(config, i % num_devices) for i, config in enumerate(configurations)],
            )
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
        finally:
            pool.close()
            pool.join()
    total_elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {timedelta(seconds=total_elapsed_time)}")
    print(
        f"Average time per experiment: {timedelta(seconds=total_elapsed_time / len(configurations))}"
    )


def main():
    args = parse_args()
    base_config = load_config(args.config, args.exp_time_str)

    # Generate hyperparameter combinations
    configurations = list(
        generate_hyperparameter_combinations(base_config, SEARCH_SPACE)
    )
    configurations = [dict_to_namespace(config) for config in configurations]

    print(f"Total number of experiments: {len(configurations)}")

    # Get the number of CUDA devices
    num_devices = torch.cuda.device_count()

    # Convert gpu_settings dictionary to argparse.Namespace
    base_config = dict_to_namespace(base_config, False)
    gpu_settings = getattr(base_config, "gpu_settings")
    gpu_settings = dict_to_namespace(gpu_settings, False)

    use_multi_gpu = getattr(gpu_settings, "use_multi_gpu", False)
    use_gpu = getattr(gpu_settings, "use_gpu", False)
    use_all_gpus_for_search = getattr(gpu_settings, "use_all_gpus_for_search", False)

    if use_multi_gpu:
        print("Using multiple GPUs for each experiment.")
        run_experiments_in_order(configurations)
    elif use_gpu:
        if use_all_gpus_for_search:
            print("Using all GPUs to conduct a grid search in parallel.")
            run_experiments_in_parallel(configurations, num_devices)
        else:
            print("Using single GPU to conduct a grid search in order.")
            run_experiments_in_order(configurations)
    else:
        raise ValueError("No GPU available for experiments.")


if __name__ == "__main__":
    main()
