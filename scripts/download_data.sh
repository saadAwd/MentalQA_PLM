#!/bin/bash
# Download and prepare MentalQA data

# Create data directory
mkdir -p data

# Check if Train_Dev.tsv exists in parent directory
if [ -f "../Train_Dev.tsv" ]; then
    echo "Found Train_Dev.tsv in parent directory"
    TSV_FILE="../Train_Dev.tsv"
else
    echo "Train_Dev.tsv not found. Please place it in the parent directory or update this script."
    exit 1
fi

# Prepare Q-types data
echo "Preparing Q-types data..."
python -m src.data_utils --tsv "$TSV_FILE" --task qtypes --output-dir data --seed 42

# Prepare A-types data
echo "Preparing A-types data..."
python -m src.data_utils --tsv "$TSV_FILE" --task atypes --output-dir data --seed 42

echo "Data preparation complete!"

