# Developer Guide: Blockchain-Enabled ML Provenance System

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Quick Start](#quick-start)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Blockchain Integration](#blockchain-integration)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)
11. [Contributing](#contributing)

## Overview

This system provides blockchain-enabled machine learning provenance tracking with support for multiple blockchain networks (IPFS, Ethereum, Bitcoin). It stores Merkle tree hashes before and after training runs to ensure data integrity and provide tamper-evident audit trails.

### Key Features

- 🔗 **Multi-Blockchain Support**: IPFS, Ethereum, Bitcoin
- 📊 **Merkle Tree Integration**: Cryptographic verification of ML pipeline
- 🔒 **Immutable Provenance**: Tamper-evident audit trails
- ⚡ **Auto Mode**: Fully automated blockchain integration
- 🛠️ **Developer Friendly**: Easy setup and configuration

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Training Pipeline                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Data       │  │   Model      │  │   Training       │  │
│  │  Provenance  │  │  Provenance  │  │  Provenance      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 ProvenanceTracker                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Merkle Tree   │  │  Blockchain     │  │  Provenance  │ │
│  │   Generation    │  │  Integration    │  │  Data        │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 BlockchainManager                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Ethereum   │  │   Bitcoin    │  │      IPFS        │  │
│  │  Interface   │  │  Interface   │  │    Interface     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Pre-Training**: Data + Model → Merkle Tree → Root Hash → Blockchain Storage
2. **Training**: Training Process → Epoch Updates → Merkle Tree Updates
3. **Post-Training**: Final Model + Results → Merkle Tree → Root Hash → Blockchain Storage
4. **Verification**: Stored Hashes → Blockchain Verification → Integrity Report

## Quick Start

### 1. Clone and Setup

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

## Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- Homebrew (for macOS)

### Dependencies

```bash
# Core ML dependencies
torch>=2.0.0
numpy==1.26.4
scikit-learn==1.4.1.post1

# Blockchain dependencies
web3>=6.0.0
requests>=2.31.0
ipfshttpclient>=0.8.0
gitpython==3.1.42

# Other dependencies
opacus==1.1.3
pandas==2.2.1
matplotlib==3.8.3
```

### Installation Steps

1. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python3 -c "from ml_provenance.provenance.blockchain import ProvenanceBlockchainTracker; print('✅ Installation successful')"
   ```

## Configuration

### Blockchain Configuration

Create or update `configs/blockchain_config.json`:

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

Update your training config to include blockchain settings:

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

## Blockchain Integration

### Supported Networks

#### 1. IPFS (InterPlanetary File System)

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

**Configuration:**
```json
{
  "ipfs": {
    "enabled": true,
    "url": "http://localhost:5001",
    "timeout": 30
  }
}
```

#### 2. Ethereum

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

**Configuration:**
```json
{
  "ethereum": {
    "enabled": true,
    "rpc_url": "http://127.0.0.1:8545",
    "private_key": "your_private_key",
    "contract_address": null
  }
}
```

#### 3. Bitcoin

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

### Smart Contract Integration

For production use, you can deploy a smart contract to store hashes:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MLProvenance {
    mapping(bytes32 => bool) public storedHashes;
    mapping(bytes32 => uint256) public timestamps;
    mapping(bytes32 => string) public metadata;
    
    event HashStored(bytes32 indexed merkleRoot, string metadata, uint256 timestamp);
    
    function storeHash(bytes32 merkleRoot, string memory metadataStr) public {
        storedHashes[merkleRoot] = true;
        timestamps[merkleRoot] = block.timestamp;
        metadata[merkleRoot] = metadataStr;
        emit HashStored(merkleRoot, metadataStr, block.timestamp);
    }
    
    function verifyHash(bytes32 merkleRoot) public view returns (bool) {
        return storedHashes[merkleRoot];
    }
    
    function getHashInfo(bytes32 merkleRoot) public view returns (bool, uint256, string memory) {
        return (storedHashes[merkleRoot], timestamps[merkleRoot], metadata[merkleRoot]);
    }
}
```

## API Reference

### ProvenanceTracker

#### Constructor
```python
ProvenanceTracker(base_dir="artifacts", config: Optional[Dict[str, Any]] = None)
```

#### Methods

##### `track_data(train_data, test_data)`
Track data provenance and generate hashes.

##### `track_model(model)`
Track model architecture and parameters.

##### `store_merkle_on_blockchain_before_training(training_config)`
Store Merkle root hash on blockchain before training begins.

**Returns:** Dictionary mapping networks to transaction IDs

##### `store_merkle_on_blockchain_after_training(training_results)`
Store Merkle root hash on blockchain after training completes.

**Returns:** Dictionary mapping networks to transaction IDs

##### `verify_blockchain_provenance()`
Verify the complete provenance chain on blockchain.

**Returns:** Dictionary containing verification results

##### `save_blockchain_report(output_path=None)`
Save blockchain report to file.

**Returns:** Path to saved report

##### `get_blockchain_status()`
Get current blockchain status and configuration.

**Returns:** Dictionary containing blockchain status

### BlockchainManager

#### Constructor
```python
BlockchainManager(config: Dict[str, Any])
```

#### Methods

##### `store_merkle_hash(merkle_root_hash, metadata, networks=None)`
Store Merkle root hash on multiple blockchain networks.

##### `verify_merkle_hash(merkle_root_hash, transaction_ids)`
Verify Merkle root hash on multiple blockchain networks.

##### `get_transaction_info(transaction_ids)`
Get transaction information from multiple blockchain networks.

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

```python
import os

config = {
    "ethereum": {
        "private_key": os.getenv("ETH_PRIVATE_KEY"),
        "rpc_url": os.getenv("ETH_RPC_URL", "http://127.0.0.1:8545")
    }
}
```

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

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/blockchain-integration
   ```
3. **Make your changes**
4. **Add tests**
5. **Update documentation**
6. **Submit a pull request**

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python scripts/demo_blockchain_provenance.py

# Run full training test
python src/ml_provenance/training/train.py
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Write unit tests

### Documentation

- Update this guide for new features
- Add examples for new functionality
- Update API documentation
- Include troubleshooting steps

---

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the API documentation
3. Check existing issues on GitHub
4. Create a new issue with detailed information

---

*This developer guide covers the blockchain-enabled ML provenance system. For more information, see the other documentation files in the `docs/` directory.* 