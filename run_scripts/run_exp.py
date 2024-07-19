import os
import sys
import time
import traceback
from datetime import timedelta

import numpy as np
import pathos.multiprocessing as mp
import torch

sys.path.append("/data3/lsf/Pein/Power-Prediction")


from run_scripts.single_exp import execute_experiment
from utils.config import dict_to_namespace, load_config, parse_args
from utils.exp_search import generate_hyperparameter_combinations

# Define the search space for hyperparameters
SEARCH_SPACE = {
    "d_model": [64, 256],
    "hidden_dim": [64, 256],
    "last_hidden_dim": [64, 512],
    "time_d_model": [64, 128, 256],
    "e_layers": [8, 16, 32],
    # from 1e-4 to 1e-1, with 50 steps
    "learning_rate": np.logspace(-4, -3, 10),
    "combine_type": ["add"],
    "train_epochs": [60],
    "seq_len": [8, 24],
}


def redirect_stdout_stderr(log_filepath):
    """Redirect stdout and stderr to a log file."""
    log_file = open(log_filepath, "w")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


def execute_experiment_with_logging(config):
    """Execute experiment with logging redirection."""
    log_filename = f"seq_len-{config.seq_len}-lr-{config.learning_rate}-d-{config.d_model}-last_d-{config.last_hidden_dim}-time_d-{config.time_d_model}"
    if config.use_multi_gpu:
        log_filename += "-allgpu.log"
    else:
        log_filename += f"-gpu-{config.gpu}.log"

    os.makedirs(config.log_output_dir, exist_ok=True)

    log_filepath = os.path.join(config.log_output_dir, log_filename)

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
        config.gpu = device_id
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
            elapsed_time = worker(config)
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
            results = pool.starmap(
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
    base_config = load_config(args.config)

    # Generate hyperparameter combinations
    configurations = list(
        generate_hyperparameter_combinations(base_config, SEARCH_SPACE)
    )
    configurations = [dict_to_namespace(config) for config in configurations]

    print(f"Total number of experiments: {len(configurations)}")

    num_devices = torch.cuda.device_count()
    use_multi_gpu = base_config.get("use_multi_gpu", False)
    use_gpu = base_config.get("use_gpu", False)
    use_all_gpus_for_search = base_config.get("use_all_gpus_for_search", False)

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
