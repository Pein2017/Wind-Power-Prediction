#!/bin/bash

# Function to convert seconds to hours, minutes, and seconds
convertsecs() {
    ((h = ${1} / 3600))
    ((m = (${1} % 3600) / 60))
    ((s = ${1} % 60))
    printf "%02d hours %02d minutes %02d seconds" $h $m $s
}

# Capture the start time
start_time=$(date +%s)

# Activate the conda environment
source activate Pein_310

# Run the first instance with GPU 0 in the background
CUDA_VISIBLE_DEVICES=0 python /data/Pein/Pytorch/Wind-Power-Prediction/run_scripts/run_optuna.py >/data/Pein/Pytorch/Wind-Power-Prediction/tmux_console_4.log 2>&1 &

# Sleep for seconds to allow the first instance to start
sleep 3

# Run the second instance with GPU 1 in the background
CUDA_VISIBLE_DEVICES=1 python /data/Pein/Pytorch/Wind-Power-Prediction/run_scripts/run_optuna.py >/data/Pein/Pytorch/Wind-Power-Prediction/tmux_console_5.log 2>&1 &

# Wait for both background processes to finish
wait

# Capture the end time
end_time=$(date +%s)

# Calculate the duration
elapsed_time=$((end_time - start_time))

# Print the total time in hours, minutes, and seconds
echo "Total time taken: $(convertsecs $elapsed_time)"

echo "Both instances have finished."
