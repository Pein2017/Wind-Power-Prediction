#!/bin/bash

# Function to convert seconds to hours, minutes, and seconds
convertsecs() {
    ((h = ${1} / 3600))
    ((m = (${1} % 3600) / 60))
    ((s = ${1} % 60))
    printf "%02d hours %02d minutes %02d seconds" $h $m $s
}


# Number of Processes
COUNT=128
START=0
END=$((START + COUNT - 1))
SLEEP=3

# Number of GPUs
GPU_COUNT=8

# Capture the start time
start_time=$(date +%s)

# Run the instances with GPUs in the background
for i in $(seq $START $END); do
    gpu_id=$((i % GPU_COUNT)) # Alternate between available GPUs
    log_file="/data/Pein/Pytorch/Wind-Power-Prediction/tmux_${i}.log"

    CUDA_VISIBLE_DEVICES=$gpu_id python /data/Pein/Pytorch/Wind-Power-Prediction/run_scripts/run_optuna.py >"$log_file" 2>&1 &

    # Sleep for seconds to allow the instance to start
    sleep $SLEEP
done

# Wait for all background processes to finish
wait

# Capture the end time
end_time=$(date +%s)

# Calculate the duration
elapsed_time=$((end_time - start_time))

# Print the total time in hours, minutes, and seconds
echo "Total time taken: $(convertsecs $elapsed_time)"

echo "All instances have finished."
