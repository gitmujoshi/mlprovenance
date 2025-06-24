# System Architecture: Blockchain-Enabled ML Provenance Tracking

## Overview

This document describes the architecture of a comprehensive machine learning provenance tracking system with blockchain integration. The system provides immutable, tamper-evident audit trails for ML training processes by storing Merkle tree hashes on multiple blockchain networks.

## High-Level Architecture

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

## Core Components

### 1. ProvenanceTracker

The main orchestrator that coordinates all provenance tracking activities.

**Key Responsibilities:**
- Initialize and manage blockchain connections
- Coordinate data, model, and training tracking
- Store Merkle root hashes on blockchain networks
- Generate and verify blockchain proofs
- Generate comprehensive reports

**Key Methods:**
```python
class ProvenanceTracker:
    def __init__(self, base_dir="artifacts", config: Optional[Dict[str, Any]] = None)
    def track_data(self, train_data, test_data)
    def track_model(self, model)
    def store_merkle_on_blockchain_before_training(self, training_config)
    def store_merkle_on_blockchain_after_training(self, training_results)
    def verify_blockchain_provenance(self)
    def save_blockchain_report(self, output_path=None)
    def get_blockchain_status(self)
```

### 2. BlockchainManager

Manages multiple blockchain network interfaces and provides unified operations.

**Key Responsibilities:**
- Initialize blockchain interfaces (Ethereum, Bitcoin, IPFS)
- Store hashes on multiple networks
- Verify hashes across networks
- Handle network-specific errors and retries

**Key Methods:**
```python
class BlockchainManager:
    def __init__(self, config: Dict[str, Any])
    def store_merkle_hash(self, merkle_root_hash, metadata, networks=None)
    def verify_merkle_hash(self, merkle_root_hash, transaction_ids)
    def get_transaction_info(self, transaction_ids)
```

### 3. Blockchain Interfaces

#### EthereumInterface
- Connects to Ethereum nodes via Web3
- Supports smart contract interactions
- Handles gas estimation and transaction signing

#### BitcoinInterface
- Connects to Bitcoin nodes via RPC
- Uses OP_RETURN for data storage
- Supports testnet and mainnet

#### IPFSInterface
- Connects to IPFS daemon via HTTP API
- Stores content-addressed data
- Provides CID-based verification

### 4. Merkle Tree System

**Components:**
- `MLProvenanceMerkleTree`: Main Merkle tree implementation
- `MerkleNode`: Individual tree nodes
- `HashFactory`: Configurable hash algorithm support

**Tree Structure:**
```
Root Hash
├── Data Node
│   ├── Training Data Hash
│   └── Test Data Hash
├── Model Node
│   ├── Architecture Hash
│   └── Weights Hash
└── Training Node
    ├── Epoch 1 Node
    │   ├── Model State Hash
    │   ├── Metrics Hash
    │   └── Privacy Metrics Hash
    ├── Epoch 2 Node
    │   ├── Model State Hash
    │   ├── Metrics Hash
    │   └── Privacy Metrics Hash
    └── ... (subsequent epochs)
```

## Data Flow

### 1. Pre-Training Phase

```
Data + Model → Merkle Tree → Root Hash → Blockchain Storage
```

1. **Data Tracking**: Generate hashes for training and test data
2. **Model Tracking**: Generate hashes for model architecture
3. **Merkle Tree Construction**: Build initial tree with data and model nodes
4. **Blockchain Storage**: Store root hash on configured networks
5. **Transaction Recording**: Store transaction IDs for verification

### 2. Training Phase

```
Training Process → Epoch Updates → Merkle Tree Updates
```

1. **Epoch Tracking**: Track metrics, model state, and privacy budget
2. **Tree Updates**: Add epoch nodes to Merkle tree
3. **Hash Generation**: Generate new root hash after each epoch
4. **Local Storage**: Store updated tree locally

### 3. Post-Training Phase

```
Final Model + Results → Merkle Tree → Root Hash → Blockchain Storage
```

1. **Final State**: Capture final model state and metrics
2. **Tree Completion**: Complete Merkle tree with all epochs
3. **Blockchain Storage**: Store final root hash on networks
4. **Verification**: Verify both pre and post-training hashes

### 4. Verification Phase

```
Stored Hashes → Blockchain Verification → Integrity Report
```

1. **Hash Retrieval**: Retrieve stored hashes from blockchain
2. **Local Verification**: Recompute hashes locally
3. **Cross-Network Verification**: Verify across multiple networks
4. **Report Generation**: Generate comprehensive verification report

## Blockchain Integration Details

### Supported Networks

#### IPFS (InterPlanetary File System)
- **Type**: Decentralized storage network
- **Storage Method**: Content-addressed storage
- **Advantages**: No fees, high availability, decentralized
- **Use Case**: Development, testing, backup storage

#### Ethereum
- **Type**: Smart contract platform
- **Storage Method**: Smart contract state
- **Advantages**: Immutable, programmable, global consensus
- **Use Case**: Production environments, regulatory compliance

#### Bitcoin
- **Type**: Cryptocurrency blockchain
- **Storage Method**: OP_RETURN transactions
- **Advantages**: Maximum security, long-term stability
- **Use Case**: High-security requirements, long-term storage

### Smart Contract Integration

For production Ethereum deployments, a smart contract can be used:

```solidity
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
}
```

## Configuration System

### Blockchain Configuration

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
      "private_key": "your_private_key",
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

## Error Handling and Resilience

### Network Failures
- Automatic retry with exponential backoff
- Fallback to local storage if blockchain unavailable
- Graceful degradation of functionality

### Data Integrity
- Hash verification before and after storage
- Cross-network verification
- Comprehensive error reporting

### Monitoring
- Blockchain status monitoring
- Transaction success tracking
- Performance metrics collection

## File Structure

```
src/ml_provenance/
├── provenance/
│   ├── blockchain.py          # Blockchain integration
│   ├── tracker.py             # Main provenance tracker
│   ├── merkle_tree.py         # Merkle tree implementation
│   ├── verifier.py            # Verification system
│   └── hash_config.py         # Hash algorithm configuration
├── training/
│   └── train.py               # Training with blockchain integration
└── utils/
    └── ...                    # Utility functions

configs/
├── blockchain_config.json     # Blockchain configuration
└── training_config_*.json     # Training configurations

scripts/
├── setup_local_geth.sh        # Local Ethereum setup
├── demo_blockchain_provenance.py  # Blockchain demo
└── ...                        # Other utility scripts
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

*This architecture provides a robust, scalable foundation for blockchain-enabled ML provenance tracking with support for multiple networks and comprehensive verification capabilities.* 