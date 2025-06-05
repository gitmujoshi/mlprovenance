#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# Verify Python interpreter
echo "Verifying Python interpreter..."
PYTHON_PATH=$(which python)
echo "Using Python interpreter: $PYTHON_PATH"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run training script
echo "Starting training..."
python "$PROJECT_ROOT/src/training/train.py"

# Deactivate virtual environment
echo -e "${GREEN}Deactivating virtual environment...${NC}"
deactivate 