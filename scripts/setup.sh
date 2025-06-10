#!/bin/bash

# Remove any existing venv to avoid conflicts
echo "Removing old virtual environment if it exists..."
rm -rf venv

# Create and activate virtual environment with python3.11
echo "Creating virtual environment with python3.11..."
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
./venv/bin/pip install --upgrade pip

# Install PyTorch first
echo "Installing PyTorch..."
./venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "Installing other requirements..."
./venv/bin/pip install -r requirements.txt

# Create data directory
echo "Creating data directory..."
mkdir -p data

# Run setup test with the venv's Python
echo "Running setup test..."
./venv/bin/python scripts/setup_test.py

# Deactivate virtual environment
deactivate

echo "Setup completed!"
echo "To use the environment, run: source venv/bin/activate OR use ./venv/bin/python <your_script.py>" 