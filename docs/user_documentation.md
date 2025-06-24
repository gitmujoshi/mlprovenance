# User Documentation: ML Provenance Tracking with Blockchain Integration

## Overview

This user documentation provides a comprehensive guide for using the ML provenance tracking system with blockchain integration. The system provides immutable, tamper-evident audit trails for machine learning training processes.

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd mnist_provenance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Local Blockchain (Optional)

```bash
# Start local Geth node for Ethereum development
bash scripts/setup_local_geth.sh

# Or use IPFS only (default)
# No additional setup required
```

### 3. Run Demo

```bash
python3 scripts/demo_blockchain_provenance.py
```

### 4. Run Training

```bash
python3 src/ml_provenance/training/train.py
```

## Features

### 🔗 Multi-Blockchain Support
- **IPFS**: Decentralized storage for development and testing
- **Ethereum**: Smart contract platform for production use
- **Bitcoin**: Maximum security for long-term storage

### 📊 Merkle Tree Integration
- Cryptographic verification of ML pipeline
- Tamper-evident audit trails
- Efficient proof generation and verification

### 🔒 Immutable Provenance
- Blockchain-based hash storage
- Timestamp verification
- Cross-network validation

### ⚡ Auto Mode
- Fully automated blockchain integration
- No manual intervention required
- Graceful fallback to local storage

## Configuration

### Blockchain Configuration

Create `configs/blockchain_config.json`:

```json
{
  "blockchain": {
    "enabled": true,
    "networks": ["ipfs", "ethereum"],
    "ipfs": {
      "enabled": true,
      "url": "http://localhost:5001",
      "timeout": 30,
      "retry_attempts": 3
    },
    "ethereum": {
      "enabled": true,
      "rpc_url": "http://127.0.0.1:8545",
      "private_key": "your_private_key_here",
      "contract_address": null,
      "gas_limit": 300000,
      "gas_price": "auto"
    },
    "storage_options": {
      "store_before_training": true,
      "store_after_training": true,
      "store_epoch_checkpoints": false
    }
  }
}
```

### Training Configuration

```python
config = {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hash_algorithm": "blake3",
    "blockchain": {
        "networks": ["ipfs", "ethereum"],
        "ipfs": {"url": "http://localhost:5001"},
        "ethereum": {
            "rpc_url": "http://127.0.0.1:8545",
            "private_key": "your_private_key"
        }
    }
}
```

## Usage Examples

### Basic Usage

```python
from ml_provenance.provenance.tracker import ProvenanceTracker
import json

# Load configuration
with open('configs/blockchain_config.json', 'r') as f:
    config = json.load(f)

# Initialize tracker with blockchain support
provenance_tracker = ProvenanceTracker(config=config)

# Track data and model
provenance_tracker.track_data(train_data, test_data)
provenance_tracker.track_model(model)

# Store pre-training hash on blockchain
training_config = {"epochs": 5, "batch_size": 32}
before_transactions = provenance_tracker.store_merkle_on_blockchain_before_training(training_config)

# ... training process ...

# Store post-training hash on blockchain
training_results = {"accuracy": 0.95, "loss": 0.1}
after_transactions = provenance_tracker.store_merkle_on_blockchain_after_training(training_results)

# Verify blockchain provenance
verification_results = provenance_tracker.verify_blockchain_provenance()

# Save reports
provenance_tracker.save_blockchain_report()
provenance_tracker.save()
```

### Advanced Usage

```python
# Get blockchain status
status = provenance_tracker.get_blockchain_status()
print(f"Blockchain enabled: {status['blockchain_enabled']}")
print(f"Stored hashes: {status['stored_hashes']}")

# Verify specific networks
verification = provenance_tracker.verify_blockchain_provenance()
if verification['chain_integrity']:
    print("✅ Provenance chain integrity verified!")
else:
    print("❌ Provenance chain integrity failed!")

# Custom blockchain configuration
custom_config = {
    "blockchain": {
        "networks": ["ipfs"],
        "ipfs": {"url": "http://custom-ipfs-node:5001"}
    }
}
provenance_tracker = ProvenanceTracker(config=custom_config)
```

## Blockchain Networks

### IPFS (InterPlanetary File System)

**Advantages:**
- Decentralized storage
- Content-addressed
- No transaction fees
- High availability

**Setup:**
```bash
# Install IPFS
brew install ipfs  # macOS
# or download from https://ipfs.io/docs/install/

# Start IPFS daemon
ipfs daemon
```

**Use Case**: Development and testing

### Ethereum

**Advantages:**
- Smart contract support
- Immutable blockchain
- Programmable verification
- Global consensus

**Setup:**
```bash
# Install Geth
brew install ethereum

# Start local dev node
bash scripts/setup_local_geth.sh
```

**Use Case**: Production environments

### Bitcoin

**Advantages:**
- Most secure blockchain
- OP_RETURN for data storage
- Global consensus
- Long-term stability

**Setup:**
```bash
# Install Bitcoin Core
brew install bitcoin

# Configure bitcoin.conf
echo "rpcuser=your_username" >> ~/.bitcoin/bitcoin.conf
echo "rpcpassword=your_password" >> ~/.bitcoin/bitcoin.conf
echo "rpcallowip=127.0.0.1" >> ~/.bitcoin/bitcoin.conf

# Start Bitcoin node
bitcoind
```

**Use Case**: High-security requirements

## Output Files

The system generates several output files:

- `blockchain_report.json` - Complete blockchain verification report
- `provenance_report.json` - Standard provenance report with blockchain info
- `merkle_tree_*.json` - Merkle tree structure files
- `geth_dev.log` - Geth development node logs

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem:** `ModuleNotFoundError: No module named 'git'`

**Solution:**
```bash
pip install gitpython
```

#### 2. Geth Connection Issues
**Problem:** `Connection refused` when connecting to Geth

**Solution:**
```bash
# Check if Geth is running
lsof -i :8545

# Start Geth if not running
bash scripts/setup_local_geth.sh
```

#### 3. IPFS Connection Issues
**Problem:** `Connection refused` when connecting to IPFS

**Solution:**
```bash
# Start IPFS daemon
ipfs daemon
```

#### 4. Private Key Issues
**Problem:** `Invalid private key` error

**Solution:**
```bash
# Extract private key from Geth dev node
echo 'eth.accounts' | geth attach http://127.0.0.1:8545
# Then extract private key from keystore file
```

#### 5. Gas Limit Issues
**Problem:** `Out of gas` error on Ethereum

**Solution:**
```json
{
  "ethereum": {
    "gas_limit": 500000,
    "gas_price": "auto"
  }
}
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configuration
config["blockchain"]["debug"] = True
```

### Network-Specific Issues

#### Ethereum
```bash
# Check Geth logs
tail -f geth_dev.log

# Check account balance
echo 'eth.getBalance(eth.accounts[0])' | geth attach http://127.0.0.1:8545
```

#### IPFS
```bash
# Check IPFS status
ipfs id

# Check if content is available
ipfs cat <CID>
```

## Best Practices

### 1. Configuration Management
- Store sensitive data (private keys) in environment variables
- Use different configurations for development and production
- Version control your configuration templates

### 2. Error Handling
```python
try:
    transactions = provenance_tracker.store_merkle_on_blockchain_before_training(config)
    if transactions:
        print("✅ Blockchain storage successful")
    else:
        print("⚠️ Blockchain storage failed")
except Exception as e:
    print(f"❌ Error: {e}")
    # Fallback to local storage only
```

### 3. Performance Optimization
- Use IPFS for development (faster, no fees)
- Use Ethereum for production (immutable, verifiable)
- Cache verification results
- Batch operations when possible

### 4. Security
- Never commit private keys to version control
- Use test networks for development
- Validate all blockchain responses
- Implement proper access controls

### 5. Monitoring
```python
# Monitor blockchain status
status = provenance_tracker.get_blockchain_status()
if not status['blockchain_enabled']:
    logger.warning("Blockchain tracking disabled")

# Monitor verification results
verification = provenance_tracker.verify_blockchain_provenance()
if not verification['chain_integrity']:
    logger.error("Provenance chain integrity failed")
```

## Demo and Testing

### Run the Demo

```bash
python3 scripts/demo_blockchain_provenance.py
```

This will:
1. Initialize a provenance tracker with blockchain support
2. Create sample data and model
3. Store pre-training hash on blockchain
4. Simulate training process
5. Store post-training hash on blockchain
6. Verify blockchain provenance
7. Generate comprehensive report

### Expected Output

```
🚀 Blockchain Provenance Demo
================================

📊 Initializing Provenance Tracker...
✅ Provenance tracker initialized with blockchain support

🔗 Blockchain Status:
   - IPFS: ✅ Connected (http://localhost:5001)
   - Ethereum: ✅ Connected (http://127.0.0.1:8545)
   - Bitcoin: ❌ Not configured

📈 Creating Sample Data and Model...
✅ Sample data created (1000 training, 200 test samples)
✅ Sample model created (2-layer neural network)

🌳 Building Merkle Tree...
✅ Merkle tree built with root hash: a1b2c3d4...

🔗 Storing Pre-Training Hash on Blockchain...
   - IPFS: ✅ Stored (CID: QmX...)
   - Ethereum: ✅ Stored (Tx: 0x...)

📊 Simulating Training Process...
   - Epoch 1: Loss=0.693, Accuracy=0.500
   - Epoch 2: Loss=0.523, Accuracy=0.750
   - Epoch 3: Loss=0.321, Accuracy=0.850

🌳 Updating Merkle Tree with Training Results...
✅ Merkle tree updated with new root hash: e5f6g7h8...

🔗 Storing Post-Training Hash on Blockchain...
   - IPFS: ✅ Stored (CID: QmY...)
   - Ethereum: ✅ Stored (Tx: 0x...)

🔍 Verifying Blockchain Provenance...
✅ Chain integrity verified across all networks
✅ All hashes match expected values
✅ Timestamps are consistent

📄 Generating Blockchain Report...
✅ Report saved to: blockchain_report.json

🎉 Demo completed successfully!
```

## Additional Resources

- **[Developer Guide](developer_guide.md)** - Comprehensive guide for developers
- **[Blockchain Integration](blockchain_provenance.md)** - Detailed blockchain documentation
- **[Architecture](ARCHITECTURE.md)** - System architecture overview
- **[Provenance Tracking](provenance_tracking.md)** - Provenance tracking concepts
- **[Model Training](model_training_and_safety.md)** - Training and safety features

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the [developer guide](developer_guide.md)
3. Check existing issues on GitHub
4. Create a new issue with detailed information

---

*This user documentation provides a comprehensive guide for using the blockchain-enabled ML provenance tracking system. For advanced usage and development, see the [Developer Guide](developer_guide.md).* 