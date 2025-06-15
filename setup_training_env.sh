#!/bin/bash
# Script to set up environment for safety_features training
set -e

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install torch transformers datasets tqdm psutil

# Install safety_features package in editable mode if setup.py exists
if [ -f safety_features/setup.py ]; then
    pip install -e safety_features/ --use-pep517
fi

# Print success message
echo "Environment setup complete. To activate, run: source venv/bin/activate" 