import os
import sys
import traceback

import pathos.multiprocessing as mp
import torch

sys.path.append("/data3/lsf/Pein/Power-Prediction")


from run_scripts.single_exp import execute_experiment
from utils.config import dict_to_namespace, load_config, parse_args
from utils.exp_search import generate_hyperparameter_combinations

# Define the search space for hyperparameters
search_space = {
    "d_model": [32],
    "hidden_dim": [64],
    "last_hidden_dim": [32],
    "time_d_model": [32],
    "seq_len": [8],
    "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "train_epochs": [100],
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
    execute_experiment_with_logging(config)


def main():
    args = parse_args()
    base_config = load_config(args.config)

    # Generate hyperparameter combinations
    configurations = list(
        generate_hyperparameter_combinations(base_config, search_space)
    )

    # Convert configurations to argparse.Namespace
    configurations = [dict_to_namespace(config) for config in configurations]

    print(f"Total number of experiments: {len(configurations)}")

    # Get the number of available devices
    num_devices = torch.cuda.device_count()

    use_multi_gpu = base_config.get("use_multi_gpu", False)
    use_gpu = base_config.get("use_gpu", False)
    use_all_gpus_for_search = base_config.get("use_all_gpus_for_search", False)

    if use_multi_gpu:
        print("Using multiple GPUs for each experiment.")
        for config in configurations:
            execute_experiment_with_logging(config)
    elif use_gpu:
        if use_all_gpus_for_search:
            print("Using all GPUs to conduct a grid search in parallel.")
            with mp.Pool(processes=num_devices) as pool:
                try:
                    pool.starmap(
                        worker,
                        [
                            (config, i % num_devices)
                            for i, config in enumerate(configurations)
                        ],
                    )
                except KeyboardInterrupt:
                    print("Caught KeyboardInterrupt, terminating workers")
                    pool.terminate()
                finally:
                    pool.close()
                    pool.join()
        else:
            print("Using single GPU to conduct a grid search in order.")
            for i, config in enumerate(configurations):
                print(f"Running {i+1}/{len(configurations)} of the experiments.")
                try:
                    worker(config)
                    print(f"Completed {i+1}/{len(configurations)} of the experiments.")
                except Exception as e:
                    print(f"Error during experiment {i+1}: {e}")
    else:
        raise ValueError("No GPU available for experiments.")


if __name__ == "__main__":
    main()
