# Blockchain-Enabled ML Provenance Tracking

## Overview

This document describes the blockchain integration features of the ML provenance tracking system. The system provides immutable, tamper-evident audit trails by storing Merkle tree hashes on multiple blockchain networks before and after training runs.

## Key Features

- **🔗 Multi-Blockchain Support**: IPFS, Ethereum, Bitcoin
- **📊 Merkle Tree Integration**: Cryptographic verification of ML pipeline
- **🔒 Immutable Provenance**: Tamper-evident audit trails
- **⚡ Auto Mode**: Fully automated blockchain integration
- **🛠️ Developer Friendly**: Easy setup and configuration

## Supported Blockchain Networks

### 1. IPFS (InterPlanetary File System)

**Type**: Decentralized storage network  
**Storage Method**: Content-addressed storage  
**Advantages**: No fees, high availability, decentralized  
**Use Case**: Development, testing, backup storage

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
    "timeout": 30,
    "retry_attempts": 3
  }
}
```

### 2. Ethereum

**Type**: Smart contract platform  
**Storage Method**: Smart contract state  
**Advantages**: Immutable, programmable, global consensus  
**Use Case**: Production environments, regulatory compliance

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
    "contract_address": null,
    "gas_limit": 300000,
    "gas_price": "auto"
  }
}
```

### 3. Bitcoin

**Type**: Cryptocurrency blockchain  
**Storage Method**: OP_RETURN transactions  
**Advantages**: Maximum security, long-term stability  
**Use Case**: High-security requirements, long-term storage

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

## System Architecture

### Data Flow

1. **Pre-Training**: Data + Model → Merkle Tree → Root Hash → Blockchain Storage
2. **Training**: Training Process → Epoch Updates → Merkle Tree Updates
3. **Post-Training**: Final Model + Results → Merkle Tree → Root Hash → Blockchain Storage
4. **Verification**: Stored Hashes → Blockchain Verification → Integrity Report

### Core Components

#### ProvenanceTracker
The main orchestrator that coordinates all provenance tracking activities.

**Key Methods:**
```python
class ProvenanceTracker:
    def store_merkle_on_blockchain_before_training(self, training_config)
    def store_merkle_on_blockchain_after_training(self, training_results)
    def verify_blockchain_provenance(self)
    def save_blockchain_report(self, output_path=None)
    def get_blockchain_status(self)
```

#### BlockchainManager
Manages multiple blockchain network interfaces and provides unified operations.

**Key Methods:**
```python
class BlockchainManager:
    def store_merkle_hash(self, merkle_root_hash, metadata, networks=None)
    def verify_merkle_hash(self, merkle_root_hash, transaction_ids)
    def get_transaction_info(self, transaction_ids)
```

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

## Smart Contract Integration

For production Ethereum deployments, you can deploy a smart contract to store hashes:

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

## Local Development Setup

### Setting Up Local Ethereum Node

1. **Install Geth:**
   ```bash
   brew install ethereum
   ```

2. **Run the setup script:**
   ```bash
   bash scripts/setup_local_geth.sh
   ```

3. **Verify the node is running:**
   ```bash
   lsof -i :8545
   ```

4. **Check account balance:**
   ```bash
   echo 'eth.getBalance(eth.accounts[0])' | geth attach http://127.0.0.1:8545
   ```

### Setting Up IPFS

1. **Install IPFS:**
   ```bash
   brew install ipfs
   ```

2. **Initialize IPFS:**
   ```bash
   ipfs init
   ```

3. **Start IPFS daemon:**
   ```bash
   ipfs daemon
   ```

4. **Verify IPFS is running:**
   ```bash
   ipfs id
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

## Security Considerations

### Private Key Management
- Store private keys in environment variables
- Use different keys for development and production
- Implement proper access controls

### Network Security
- Use HTTPS for RPC endpoints
- Validate blockchain responses
- Implement retry mechanisms with exponential backoff

### Data Privacy
- Only store hashes, not raw data
- Consider metadata sensitivity
- Implement access controls for blockchain data

## Performance Considerations

### Network Selection
- **IPFS**: Fastest, no fees, good for development
- **Ethereum**: Medium speed, gas fees, production-ready
- **Bitcoin**: Slowest, low fees, maximum security

### Optimization Strategies
- Cache verification results
- Batch operations when possible
- Use appropriate gas limits for Ethereum
- Implement connection pooling

## Output Files

The system generates several output files:

- `blockchain_report.json` - Complete blockchain verification report
- `provenance_report.json` - Standard provenance report with blockchain info
- `merkle_tree_*.json` - Merkle tree structure files
- `geth_dev.log` - Geth development node logs

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

## Future Enhancements

### Planned Features
- **Multi-signature support**: Require multiple signatures for critical operations
- **Time-locked contracts**: Automatic verification at specific intervals
- **Cross-chain verification**: Verify hashes across different blockchain networks
- **Zero-knowledge proofs**: Privacy-preserving verification
- **Automated compliance**: Regulatory compliance reporting

### Integration Opportunities
- **CI/CD pipelines**: Automated blockchain verification in deployment
- **Model registries**: Integration with ML model registries
- **Audit systems**: Integration with external audit systems
- **Legal frameworks**: Compliance with data governance regulations

---

*This document provides comprehensive guidance for using blockchain-enabled ML provenance tracking. For more information, see the [Developer Guide](developer_guide.md) and [Architecture Documentation](ARCHITECTURE.md).* 