import itertools
import os
import subprocess
import time

# Define the lists of parameters
# d_model_list = [32, 64, 128, 256]
# hidden_dim_list = [64, 128, 256]
# last_hidden_dim_list = [128, 256, 512]
# time_d_model_list = [32, 128, 256]
# seq_len_list = [28]
# learning_rate_list = [0.3, 0.4, 0.5, 0.6]


d_model_list = [32]
time_d_model_list = [256]
hidden_dim_list = [128]
last_hidden_dim_list = [128]
seq_len_list = [28]
learning_rate_list = [0.1]

# Other fixed parameters
model_name = "SimpleMLP"
train_epochs = 5
down_sampling_layers = 2
down_sampling_window = 2
e_layers = 8

combine_type = "add"
batch_size = 1100
input_dim = 75
dec_in = 75
output_dim = 1
pred_len = 1
data_root_dir = "/data3/lsf/Pein/Power-Prediction/data"
train_path = "train_data_89_withTime.csv"
test_path = "test_data_89_withTime.csv"
scheduler_type = "OneCycleLR"
scale_x_type = "standard"
scale_y_type = "standard"
use_multi_gpu = "False"
gpu = 0
data = "WindPower"
output_dir = "/data3/lsf/Pein/Power-Prediction/output/debug"
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = "/data3/lsf/Pein/Power-Prediction/output/checkpoint"
tb_log_dir = "/data3/lsf/Pein/Power-Prediction/output/tb_log"


def build_command(search_params, gpu_id):
    d_model, hidden_dim, last_hidden_dim, time_d_model, learning_rate, seq_len = (
        search_params
    )
    comment = f"mlp_d_model_{d_model}_hidden_{hidden_dim}_timed_{time_d_model}_last_{last_hidden_dim}"

    cmd = [
        "/home/lsf/anaconda3/envs/Pein_310/bin/python",
        "/data3/lsf/Pein/Power-Prediction/exp/train.py",
        "--task_name",
        "WindPower",
        "--tb_log_dir",
        tb_log_dir,
        "--checkpoint_dir",
        checkpoint_dir,
        "--scheduler_type",
        scheduler_type,
        "--combine_type",
        combine_type,
        "--last_hidden_dim",
        str(last_hidden_dim),
        "--time_d_model",
        str(time_d_model),
        "--is_training",
        "1",
        "--scale_x_type",
        str(scale_x_type),
        "--scale_y_type",
        str(scale_y_type),
        "--output_dir",
        output_dir,
        "--comment",
        comment,
        "--loss",
        "mse",
        "--input_dim",
        str(input_dim),
        "--dec_in",
        str(dec_in),
        "--output_dim",
        str(output_dim),
        "--data_root_dir",
        data_root_dir,
        "--train_path",
        train_path,
        "--test_path",
        test_path,
        "--model",
        str(model_name),
        "--data",
        data,
        "--seq_len",
        str(seq_len),
        "--target",
        "power",
        "--e_layers",
        str(e_layers),
        "--des",
        "CombinedEmb",
        "--itr",
        "1",
        "--d_model",
        str(d_model),
        "--hidden_dim",
        str(hidden_dim),
        "--batch_size",
        str(batch_size),
        "--train_epochs",
        str(train_epochs),
        "--learning_rate",
        str(learning_rate),
        "--down_sampling_layers",
        str(down_sampling_layers),
        "--down_sampling_method",
        "avg",
        "--down_sampling_window",
        str(down_sampling_window),
        "--use_gpu",
        "True",
        "--gpu",
        str(gpu_id),
        "--use_multi_gpu",
        str(use_multi_gpu),
    ]

    log_file = os.path.join(
        output_dir, f"seq_len-{seq_len}-lr-{learning_rate}gpu-{gpu_id}.log"
    )
    return cmd, log_file


def run_experiment(cmd, log_file):
    with open(log_file, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=log)


def main():
    # Create combinations of parameters
    param_combinations = list(
        itertools.product(
            d_model_list,
            hidden_dim_list,
            last_hidden_dim_list,
            time_d_model_list,
            learning_rate_list,
            seq_len_list,
        )
    )
    total_combinations = len(param_combinations)
    gpu_processes = [None] * 2

    start_time = time.time()
    for i, params in enumerate(param_combinations):
        # Find the next available gpu
        for gpu_id in range(8):
            if (
                gpu_processes[gpu_id] is None
                or gpu_processes[gpu_id].poll() is not None
            ):
                cmd, log_file = build_command(params, gpu_id)
                gpu_processes[gpu_id] = subprocess.Popen(
                    cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT
                )
                break
        else:
            # If no gpu is available, wait for the first one to finish
            first_to_finish = min(gpu_processes, key=lambda p: p.wait())
            gpu_id = gpu_processes.index(first_to_finish)
            cmd, log_file = build_command(params, gpu_id)
            gpu_processes[gpu_id] = subprocess.Popen(
                cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT
            )

        # Calculate progress
        elapsed_time = time.time() - start_time
        progress = (i + 1) / total_combinations
        time_left = (elapsed_time / progress) * (1 - progress)

        # Print progress every 5 iterations
        if i % 5 == 0 or i == total_combinations - 1:
            print(f"\nProgress: {progress * 100:.2f}%")
            print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
            print(f"Estimated time left: {time_left / 60:.2f} minutes")

    # Wait for all processes to complete
    for process in gpu_processes:
        if process is not None:
            process.wait()


if __name__ == "__main__":
    main()
