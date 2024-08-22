#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

BASE_DIR='/data/Pein/Pytorch/Wind-Power-Prediction'
VIRTUAL_NAME="Pein_310"  
PYTHON_VERSION="3.10"    

echo "Starting server setup for $VIRTUAL_NAME..."

# Source the Conda script to enable the 'conda' command
source /opt/conda/bin/activate

# Create and activate the Conda environment
echo "Creating and activating the Conda environment..."
conda create -n ${VIRTUAL_NAME} python=${PYTHON_VERSION} -y
source /opt/conda/bin/activate ${VIRTUAL_NAME}

# Install server-specific dependencies
echo "Installing server dependencies..."
pip install --no-cache-dir -r ${BASE_DIR}/requirements.txt

echo "Finished..."

# # Run the Python server
# echo "Running the Python server..."
# cd /app/${VIRTUAL_NAME}
# python server/${VIRTUAL_NAME}_server.py
