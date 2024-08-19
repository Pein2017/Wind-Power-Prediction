#!/bin/bash

# Find and kill all processes created by run_optuna_parallel.sh
# Change the script name as needed to match the exact script name or pattern

# Get the PIDs of the processes
pids=$(ps aux | grep '/data/Pein/Pytorch/Wind-Power-Prediction/run_scripts/run_optuna.py' | grep -v grep | awk '{print $2}')

# Check if there are any processes to kill
if [ -z "$pids" ]; then
  echo "No processes found to kill."
else
  echo "Killing the following processes: $pids"
  # Kill the processes
  kill $pids
fi

# Optional: Check if the processes were successfully killed
pids=$(ps aux | grep '/data/Pein/Pytorch/Wind-Power-Prediction/run_scripts/run_optuna.py' | grep -v grep | awk '{print $2}')
if [ -z "$pids" ]; then
  echo "All processes killed successfully."
else
  echo "Some processes could not be killed: $pids"
fi
