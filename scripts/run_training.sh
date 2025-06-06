#!/bin/bash

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify Python interpreter
echo "Verifying Python interpreter..."
which python

# Start training
echo "Starting training..."
python src/training/train.py 