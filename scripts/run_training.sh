#!/bin/bash

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Create necessary directories
mkdir -p artifacts/provenance
mkdir -p artifacts/models

# Run training
echo "Starting training..."
python -m src.training.train

# Print completion message
echo "Training completed. Check artifacts/provenance for the latest run." 