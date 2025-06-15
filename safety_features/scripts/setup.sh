#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for training with safety features..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install torch transformers datasets tqdm requests

# Add safety_features to PYTHONPATH
echo "Adding safety_features to PYTHONPATH..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install safety_features package in development mode
echo "Installing safety_features package in development mode..."
pip install -e safety_features/ --use-pep517

echo "Setup complete! You can now run the training script with:"
echo "./venv/bin/python safety_features/scripts/train_gpt2_with_safety.py --sample-size 100 --epochs 1 --batch-size 8 --min-age-rating TEEN --max-input-length 512 --content-filters \"violence\" \"explicit\" \"offensive\" --block-sensitive-topics --require-content-warning"