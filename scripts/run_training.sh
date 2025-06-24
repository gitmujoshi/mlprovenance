#!/bin/bash

# Exit on error
set -e

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Run the training script
echo "Running training script..."
python -m ml_provenance.training.train

# Deactivate virtual environment
deactivate

# Create necessary directories
mkdir -p artifacts/provenance

# Print completion message
echo "Training completed. Check artifacts/provenance for the latest run." 