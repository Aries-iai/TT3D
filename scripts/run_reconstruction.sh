#!/bin/bash

# Set the script to exit immediately if any command returns a non-zero status
set -e

# Check if sufficient arguments were provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <metadata_path> <workspace_path>"
    exit 1
fi

# Assign the command line arguments to variables
METADATA_PATH="$1"
WORKSPACE_PATH="$2"

# Extract the object name from the metadata path
OBJECT_NAME=$(basename $METADATA_PATH)

# Create the directory for logs if it doesn't already exist
mkdir -p logs/reconstruction

# 3D Object Reconstruction Phase 1
echo "Starting 3D object reconstruction phase 1 for $OBJECT_NAME..."
python reconstruction.py \
    $METADATA_PATH \
    --workspace $WORKSPACE_PATH \
    -O \
    --bound 1 \
    --scale 0.8 \
    --dt_gamma 0 \
    --stage 0 \
    --lambda_tv 1e-8 > logs/reconstruction/${OBJECT_NAME}_phase_0.log 2>&1
echo "3D object reconstruction phase 1 for $OBJECT_NAME completed."

# Wait for 1 second to ensure the first phase has fully ended
sleep 1

# 3D Object Reconstruction Phase 2
echo "Starting 3D object reconstruction phase 2 for $OBJECT_NAME..."
python reconstruction.py \
    $METADATA_PATH/ \
    --workspace $WORKSPACE_PATH \
    -O \
    --bound 1 \
    --scale 0.8 \
    --dt_gamma 0 \
    --stage 1 > logs/reconstruction/${OBJECT_NAME}_phase_1.log 2>&1
echo "3D object reconstruction phase 2 for $OBJECT_NAME completed."
