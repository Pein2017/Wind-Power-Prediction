#!/bin/bash

# Define the folder name as a variable
FOLDER_NAME="24-08-16-mlp_v3-test"

# Define the base directory path
BASE_DIR="/data3/lsf/Pein/Power-Prediction"

# Remove the three folders using the folder name variable
rm -rf "$BASE_DIR/res_output/$FOLDER_NAME"
rm -rf "$BASE_DIR/optuna_results/$FOLDER_NAME"
rm -rf "$BASE_DIR/train_log/$FOLDER_NAME"

# Print a confirmation message
echo "Removed folders related to $FOLDER_NAME."
