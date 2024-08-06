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
OBJECT_NAME=$(basename "$METADATA_PATH")

# Ensure the logging directory exists
mkdir -p logs/evaluation

# Start the evaluation process
echo "Starting evaluation for $OBJECT_NAME..."
python evaluation.py \
    "$METADATA_PATH" \
    --workspace "$WORKSPACE_PATH" \
    --evaluation_model resnet \
    -O \
    --bound 1 \
    --scale 0.8 \
    --dt_gamma 0 \
    --stage 1 > logs/evaluation/${OBJECT_NAME}_evaluation.log 2>&1
echo "Evaluation completed for $OBJECT_NAME."

# Optionally, review the log or additional processing
echo "Review the logs at logs/evaluation/${OBJECT_NAME}_evaluation.log"
