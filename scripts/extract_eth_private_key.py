import json
from eth_account import Account
from pathlib import Path

# Path to your keystore file
keystore_dir = Path.home() / ".ethereum" / "keystore"
keystore_files = list(keystore_dir.glob("UTC--*"))

if not keystore_files:
    print("No keystore files found in", keystore_dir)
    exit(1)

keystore_path = keystore_files[0]
print(f"Using keystore file: {keystore_path}")

# Use the known password
password = "password123"

with open(keystore_path, "r") as f:
    keystore = json.load(f)

try:
    private_key = Account.decrypt(keystore, password)
    print("Your private key (hex):", private_key.hex())
except Exception as e:
    print("Failed to decrypt keystore:", e)
    exit(1) 