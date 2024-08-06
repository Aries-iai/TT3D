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

# Ensure the logging directory exists
mkdir -p logs/adv_optimization

# 3D Adversarial Optimization
echo "Starting 3D adversarial optimization for $OBJECT_NAME..."
python generate_3d_adv.py \
    $METADATA_PATH/ \
    --workspace $WORKSPACE_PATH \
    --target_label random \
    --surrogate_model resnet \
    -O \
    --bound 1 \
    --scale 0.8 \
    --dt_gamma 0 \
    --lambda_lap 1e-3 \
    --lambda_cd 3000 \
    --lambda_edgelen 1e-2 \
    --stage 1 > logs/adv_optimization/${OBJECT_NAME}_temp.log 2>&1
echo "3D adversarial optimization completed."

# Optionally, process the log to extract the target label and number, and rename the log file
# Extracting the full target information from the line 'We are attacking object towards <number>: <label>'
TARGET_INFO=$(grep 'We are attacking object towards' logs/adv_optimization/${OBJECT_NAME}_temp.log | awk -F 'towards ' '{print $2}' | tr -d '\n')
if [ -z "$TARGET_INFO" ]; then
    TARGET_INFO="unknown:unknown"
fi

# Rename the temporary log file to include both the target number and label
mv logs/adv_optimization/${OBJECT_NAME}_temp.log logs/adv_optimization/${OBJECT_NAME}_${TARGET_INFO}.log