#!/bin/bash
# This script extracts the Ethereum private key from the Geth keystore and runs the training with it.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Extract the private key using Python from virtual environment
ETH_PRIVATE_KEY=$("$PROJECT_ROOT/venv/bin/python" "$SCRIPT_DIR/extract_eth_private_key.py" | grep 'Your private key (hex):' | awk '{print $5}')

if [ -z "$ETH_PRIVATE_KEY" ]; then
  echo "Failed to extract Ethereum private key."
  exit 1
fi

echo "Extracted ETH_PRIVATE_KEY: $ETH_PRIVATE_KEY"

# Run the training with the extracted private key
ETH_PRIVATE_KEY=$ETH_PRIVATE_KEY "$PROJECT_ROOT/venv/bin/python" "$PROJECT_ROOT/src/ml_provenance/training/train.py"

# Run blockchain verification after training
echo ""
echo "🔍 Running blockchain verification..."
"$PROJECT_ROOT/venv/bin/python" "$SCRIPT_DIR/check_blockchain.py" 