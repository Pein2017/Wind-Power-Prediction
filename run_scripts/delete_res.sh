#!/bin/bash

# Define the folder name as a variable
FOLDER_NAME="24-08-25-hid_d-search"

# Define the base directory path
BASE_DIR="/data/Pein/Pytorch/Wind-Power-Prediction"

# Remove the three folders using the folder name variable
rm -rf "$BASE_DIR/res_output/$FOLDER_NAME"
rm -rf "$BASE_DIR/optuna_results/$FOLDER_NAME"
rm -rf "$BASE_DIR/train_log/$FOLDER_NAME"
rm -rf "$BASE_DIR/final_best_metric/$FOLDER_NAME.log"

# Print a confirmation message
echo "Removed folders related to $FOLDER_NAME."
