#!/bin/bash
# Script to set up and run a local Geth (Go Ethereum) node in dev mode
# Prints the default account and private key for use in blockchain config

set -e

# Check if geth is installed
if ! command -v geth &> /dev/null; then
    echo "[INFO] Geth not found. Installing via Homebrew..."
    brew install geth
else
    echo "[INFO] Geth is already installed."
fi

# Set data directory for dev chain
GETH_DATA_DIR="./.geth_dev_data"

# Kill any running geth dev node
if pgrep -f "geth --dev" > /dev/null; then
    echo "[INFO] Stopping existing geth dev node..."
    pkill -f "geth --dev"
    sleep 2
fi

# Remove old data dir (optional)
# rm -rf "$GETH_DATA_DIR"

# Start geth in dev mode (background)
echo "[INFO] Starting geth in dev mode..."
nohup geth --dev --http --http.api personal,eth,net,web3,miner --http.addr 127.0.0.1 --http.port 8545 --datadir "$GETH_DATA_DIR" > geth_dev.log 2>&1 &

sleep 5

# Get the default account address
ACCOUNT=$(geth attach http://127.0.0.1:8545 --exec "eth.accounts[0]")
echo "[INFO] Default dev account: $ACCOUNT"

# Extract the private key (for dev mode, it's stored in the keystore)
KEYSTORE_FILE=$(find "$GETH_DATA_DIR"/keystore -type f | head -n 1)
if [ -z "$KEYSTORE_FILE" ]; then
    echo "[ERROR] Could not find keystore file."
    exit 1
fi

# Use python to extract the private key (password is empty in dev mode)
PRIVATE_KEY=$(python3 -c "import json; from eth_keyfile import decode_keyfile_json; f=open('$KEYSTORE_FILE'); k=json.load(f); print(decode_keyfile_json(k, b'' ).hex())")
echo "[INFO] Private key for dev account: $PRIVATE_KEY"

echo ""
echo "[INFO] Update your configs/blockchain_config.json as follows:"
echo "  \"ethereum\": {"
echo "    \"enabled\": true,"
echo "    \"rpc_url\": \"http://127.0.0.1:8545\"," 
echo "    \"private_key\": \"$PRIVATE_KEY\"," 
echo "    \"contract_address\": null"
echo "  }"
echo ""
echo "[INFO] Geth dev node is running. To stop it: pkill -f 'geth --dev'" 