# MNIST Provenance Project - Architecture and Design Document

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Blockchain Integration](#blockchain-integration)
5. [Provenance Tracking System](#provenance-tracking-system)
6. [Security Architecture](#security-architecture)
7. [Data Flow](#data-flow)
8. [Integration Points](#integration-points)
9. [Deployment Architecture](#deployment-architecture)
10. [Performance Considerations](#performance-considerations)
11. [Monitoring and Logging](#monitoring-and-logging)
12. [Testing Strategy](#testing-strategy)

## Project Overview

The MNIST Provenance Project is a comprehensive machine learning system that implements end-to-end provenance tracking with blockchain integration. The system ensures data integrity, model reproducibility, and auditability throughout the ML lifecycle.

### Key Features
- **Provenance Tracking**: Complete audit trail of data, model, and training artifacts
- **Blockchain Integration**: Immutable storage of provenance hashes on Ethereum
- **Privacy-Preserving Training**: Differential privacy with Opacus
- **Multi-Hash Support**: Configurable hash algorithms (BLAKE3, SHA256, SHA512)
- **Merkle Tree Verification**: Cryptographic proof of data integrity
- **Security Best Practices**: Environment-based key management

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    MNIST Provenance System                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Training  │  │  Provenance │  │ Blockchain  │             │
│  │   Engine    │  │   Tracker   │  │ Integration │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Data      │  │   Model     │  │ Verification│             │
│  │   Manager   │  │   Manager   │  │   Engine    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Ethereum  │  │     IPFS    │  │   Bitcoin   │             │
│  │   Network   │  │   Storage   │  │   Network   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow
```
Training Request
       ↓
┌─────────────────┐
│  Data Loading   │ → MNIST Dataset
└─────────────────┘
       ↓
┌─────────────────┐
│ Provenance Init │ → Hash Generation
└─────────────────┘
       ↓
┌─────────────────┐
│ Model Creation  │ → Architecture Tracking
└─────────────────┘
       ↓
┌─────────────────┐
│ Pre-Training    │ → Merkle Root → Blockchain
└─────────────────┘
       ↓
┌─────────────────┐
│ Training Loop   │ → Epoch Tracking
└─────────────────┘
       ↓
┌─────────────────┐
│ Post-Training   │ → Final Merkle → Blockchain
└─────────────────┘
       ↓
┌─────────────────┐
│ Verification    │ → Integrity Check
└─────────────────┘
```

## Core Components

### 1. Training Engine (`src/ml_provenance/training/train.py`)
**Purpose**: Orchestrates the complete ML training pipeline with provenance tracking.

**Key Responsibilities**:
- Dataset loading and preprocessing
- Model initialization and training
- Integration with provenance tracking
- Blockchain transaction management
- Verification and reporting

**Key Classes**:
- `main()`: Entry point and orchestration
- `train_model()`: Core training loop
- `verify_blockchain_report()`: Blockchain verification

### 2. Provenance Tracker (`src/ml_provenance/provenance/tracker.py`)
**Purpose**: Manages the complete provenance lifecycle and artifact tracking.

**Key Responsibilities**:
- Data provenance tracking
- Model architecture tracking
- Training process tracking
- Merkle tree construction
- Blockchain integration

**Key Classes**:
- `ProvenanceTracker`: Main tracker class
- `Tracker`: Legacy tracker for backward compatibility

### 3. Blockchain Integration (`src/ml_provenance/provenance/blockchain.py`)
**Purpose**: Provides blockchain interfaces for immutable provenance storage.

**Key Responsibilities**:
- Ethereum transaction management
- IPFS storage integration
- Bitcoin OP_RETURN support
- Transaction verification
- Multi-network support

**Key Classes**:
- `BlockchainManager`: Multi-network blockchain manager
- `EthereumInterface`: Ethereum-specific operations
- `IPFSInterface`: IPFS storage operations
- `BitcoinInterface`: Bitcoin OP_RETURN operations
- `ProvenanceBlockchainTracker`: High-level blockchain tracking

### 4. Hash Configuration (`src/ml_provenance/provenance/hash_config.py`)
**Purpose**: Manages cryptographic hash algorithms and configurations.

**Key Responsibilities**:
- Hash algorithm selection
- Hash factory management
- Algorithm performance optimization
- Security level configuration

**Key Classes**:
- `HashFactory`: Hash algorithm factory
- `TrainingHashConfig`: Training-specific hash configuration

### 5. Merkle Tree (`src/ml_provenance/provenance/merkle_tree.py`)
**Purpose**: Implements Merkle tree data structure for cryptographic verification.

**Key Responsibilities**:
- Merkle tree construction
- Hash aggregation
- Proof generation
- Verification support

**Key Classes**:
- `MLProvenanceMerkleTree`: ML-specific Merkle tree implementation

### 6. Verification Engine (`src/ml_provenance/provenance/verifier.py`)
**Purpose**: Verifies provenance integrity and generates verification reports.

**Key Responsibilities**:
- Component verification
- Hash validation
- Report generation
- Integrity checking

**Key Classes**:
- `ProvenanceVerifier`: Main verification engine

## Blockchain Integration

### Ethereum Integration
**Purpose**: Store provenance hashes on Ethereum blockchain for immutability.

**Components**:
- `EthereumInterface`: Handles Ethereum transactions
- Web3.py integration for transaction signing
- Gas estimation and transaction management
- Error handling and fallback mechanisms

**Transaction Flow**:
```
1. Merkle Root Generation
2. Transaction Construction
3. Private Key Signing
4. Transaction Broadcasting
5. Confirmation Tracking
```

**Security Features**:
- Environment variable key management
- Transaction signing with private keys
- Gas limit and price management
- Error handling and simulation fallback

#### Detailed Ethereum Implementation

**EthereumInterface Class**:
The `EthereumInterface` class provides comprehensive Ethereum blockchain integration with the following key features:

**Initialization and Configuration**:
```python
class EthereumInterface(BlockchainInterface):
    def __init__(self, config: Dict[str, Any]):
        # Initialize Web3 connection
        self.web3 = None
        self.private_key = None
        self.account_address = None
        
        # Get private key from environment or config
        self.private_key = config.get('private_key') or os.getenv('ETH_PRIVATE_KEY')
        
        # Initialize Web3 connection
        rpc_url = config.get('rpc_url', 'http://127.0.0.1:8545')
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Get account address from private key
        account = self.web3.eth.account.from_key(self.private_key)
        self.account_address = account.address
```

**Transaction Construction and Signing**:
```python
def store_hash(self, merkle_root_hash: str, metadata: Dict[str, Any]) -> str:
    # Prepare transaction data
    transaction_data = {
        'merkle_root_hash': merkle_root_hash,
        'timestamp': int(time.time()),
        'metadata': json.dumps(metadata),
        'block_number': self.web3.eth.block_number
    }
    
    # Convert to hex string for transaction data
    data_hex = json.dumps(transaction_data).encode('utf-8').hex()
    
    # Get current gas price and estimate gas limit
    gas_price = self.web3.eth.gas_price
    gas_limit = self.config.get('gas_limit', 300000)
    
    # Build transaction
    transaction = {
        'nonce': self.web3.eth.get_transaction_count(self.account_address),
        'to': self.account_address,  # Send to self (OP_RETURN equivalent)
        'value': 0,
        'gas': gas_limit,
        'gasPrice': gas_price,
        'data': f"0x{data_hex}",
        'chainId': self.web3.eth.chain_id
    }
    
    # Sign transaction
    signed_txn = self.web3.eth.account.sign_transaction(transaction, self.private_key)
    
    # Send transaction
    tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
    
    # Wait for confirmation
    tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    
    return tx_hash.hex()
```

**Error Handling and Fallback**:
```python
try:
    # Real transaction implementation
    tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
    return tx_hash.hex()
except Exception as e:
    logger.error(f"Failed to store hash on Ethereum: {e}")
    # Fallback to simulation
    transaction_id = self._simulate_transaction(transaction_data)
    return transaction_id
```

**Transaction Verification**:
```python
def verify_hash(self, transaction_id: str, expected_hash: str) -> bool:
    # Get transaction receipt
    tx_receipt = self.web3.eth.get_transaction_receipt(transaction_id)
    
    if tx_receipt and tx_receipt.status == 1:
        # Transaction was successful
        return True
    else:
        # Transaction failed or not found
        return False
```

**Network Status Monitoring**:
```python
def get_status(self) -> Dict[str, Any]:
    status = super().get_status()
    
    if self.web3:
        try:
            latest_block = self.web3.eth.block_number
            status.update({
                'connected': True,
                'status': 'connected',
                'latest_block': latest_block,
                'account_address': self.account_address,
                'private_key_set': bool(self.private_key)
            })
        except Exception as e:
            status.update({
                'connected': False,
                'status': f'error: {str(e)}',
                'private_key_set': bool(self.private_key)
            })
    
    return status
```

#### Configuration Parameters

**Ethereum Configuration**:
```json
{
  "ethereum": {
    "enabled": true,
    "rpc_url": "http://127.0.0.1:8545",
    "private_key": null,
    "contract_address": null,
    "gas_limit": 300000,
    "gas_price": "auto"
  }
}
```

**Configuration Options**:
- `enabled`: Enable/disable Ethereum integration
- `rpc_url`: Ethereum RPC endpoint (local Geth, Infura, etc.)
- `private_key`: Private key for transaction signing (use environment variable)
- `contract_address`: Smart contract address (optional)
- `gas_limit`: Maximum gas for transactions
- `gas_price`: Gas price strategy ("auto" for dynamic pricing)

#### Security Implementation

**Private Key Management**:
```python
# Secure key loading from environment
private_key = os.getenv('ETH_PRIVATE_KEY')
if not private_key:
    raise SecurityError("Private key not found in environment")

# Remove '0x' prefix if present
if private_key.startswith('0x'):
    private_key = private_key[2:]
```

**Transaction Security**:
- **Gas Limit Protection**: Prevents excessive gas consumption
- **Nonce Management**: Ensures transaction ordering
- **Chain ID Validation**: Prevents replay attacks
- **Transaction Signing**: Secure private key signing

#### Error Handling and Recovery

**Network Failures**:
```python
# Handle network connectivity issues
try:
    self.web3 = Web3(Web3.HTTPProvider(rpc_url))
except Exception as e:
    logger.error(f"Failed to connect to Ethereum: {e}")
    self.web3 = None
```

**Transaction Failures**:
```python
# Handle transaction failures with fallback
try:
    tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
except Exception as e:
    logger.error(f"Transaction failed: {e}")
    # Fallback to simulation mode
    return self._simulate_transaction(transaction_data)
```

**Simulation Mode**:
```python
def _simulate_transaction(self, transaction_data: Dict[str, Any]) -> str:
    # Create simulated transaction ID for testing
    data_str = json.dumps(transaction_data, sort_keys=True)
    simulated_hash = hashlib.sha256(data_str.encode()).hexdigest()
    return f"simulated_ethereum_tx_{simulated_hash[:16]}"
```

#### Integration with Provenance System

**Blockchain Manager Integration**:
```python
class BlockchainManager:
    def _initialize_interfaces(self):
        # Initialize Ethereum interface
        if 'ethereum' in blockchain_config:
            eth_config = blockchain_config['ethereum']
            if eth_config.get('enabled', False):
                private_key = eth_config.get('private_key')
                env_private_key = os.getenv('ETH_PRIVATE_KEY')
                
                if private_key or env_private_key:
                    self.interfaces['ethereum'] = EthereumInterface(eth_config)
```

**Provenance Storage**:
```python
def store_provenance_on_blockchain(self, provenance_data: Dict[str, Any]) -> Dict[str, Any]:
    # Create Merkle root hash from provenance data
    merkle_tree = MLProvenanceMerkleTree()
    merkle_root = merkle_tree.build_tree(provenance_data)
    
    # Store on blockchain networks
    transactions = self.blockchain_manager.store_merkle_hash(
        merkle_root, metadata
    )
    
    return {
        'success': True,
        'merkle_root': merkle_root,
        'transactions': transactions,
        'timestamp': time.time()
    }
```

#### Performance Optimization

**Gas Optimization**:
- Dynamic gas price estimation
- Configurable gas limits
- Transaction batching support
- Network congestion handling

**Transaction Monitoring**:
- Real-time transaction status
- Confirmation tracking
- Block number monitoring
- Network health checks

#### Development and Testing

**Local Development Setup**:
```bash
# Start local Geth node
geth --dev --http --http.api personal,eth,net,web3,miner

# Set environment variable
export ETH_PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000001

# Run training with blockchain integration
python src/ml_provenance/training/train.py
```

**Testing Configuration**:
```python
# Test configuration for development
ethereum_config = {
    "enabled": True,
    "rpc_url": "http://127.0.0.1:8545",
    "private_key": None,  # Use environment variable
    "gas_limit": 300000,
    "gas_price": "auto"
}
```

#### Production Deployment

**Production Considerations**:
- **Network Selection**: Mainnet vs testnet
- **Gas Management**: Cost optimization strategies
- **Security**: Hardware security modules (HSM)
- **Monitoring**: Transaction monitoring and alerting
- **Backup**: Multiple RPC endpoints

**Environment Variables**:
```bash
# Production environment variables
export ETH_PRIVATE_KEY=your_secure_private_key
export ETH_RPC_URL=https://mainnet.infura.io/v3/your_project_id
export ETH_GAS_LIMIT=500000
export ETH_GAS_PRICE=auto
```

### IPFS Integration
**Purpose**: Decentralized storage of provenance metadata.

**Components**:
- `IPFSInterface`: IPFS storage operations
- Content-addressed storage
- Metadata management
- Retrieval and verification

### Bitcoin Integration
**Purpose**: Store provenance hashes using Bitcoin OP_RETURN.

**Components**:
- `BitcoinInterface`: Bitcoin transaction operations
- OP_RETURN data encoding
- Transaction broadcasting
- Verification support

## Provenance Tracking System

### Data Provenance
**Tracking Elements**:
- Dataset statistics (mean, std, min, max)
- Data shape and type information
- Hash of training and test data
- Metadata and timestamps

**Implementation**:
```python
def track_data(self, train_data, test_data):
    # Calculate statistics
    # Generate hashes
    # Store metadata
    # Update Merkle tree
```

### Model Provenance
**Tracking Elements**:
- Model architecture
- Layer configurations
- Parameter counts
- Model hashes

**Implementation**:
```python
def track_model(self, model):
    # Extract architecture
    # Calculate parameters
    # Generate model hash
    # Store metadata
```

### Training Provenance
**Tracking Elements**:
- Training configuration
- Hyperparameters
- Training metrics
- Privacy parameters
- Epoch-level tracking

**Implementation**:
```python
def track_training(self, model, config, final_metrics):
    # Track configuration
    # Monitor metrics
    # Store privacy parameters
    # Update Merkle tree
```

## Security Architecture

### Private Key Management
**Security Principles**:
- Never store private keys in code
- Use environment variables
- Implement key rotation
- Secure key storage

**Implementation**:
```python
# Secure key loading
private_key = os.getenv('ETH_PRIVATE_KEY')
if not private_key:
    raise SecurityError("Private key not found in environment")
```

### Hash Algorithm Security
**Security Features**:
- Configurable hash algorithms
- Cryptographic strength validation
- Performance optimization
- Collision resistance

**Supported Algorithms**:
- BLAKE3 (recommended)
- SHA256
- SHA512

### Data Integrity
**Protection Mechanisms**:
- Merkle tree verification
- Hash chain validation
- Blockchain immutability
- Cross-network verification

## Data Flow

### Training Pipeline Flow
```
1. Configuration Loading
   ↓
2. Data Loading & Preprocessing
   ↓
3. Provenance Initialization
   ↓
4. Model Creation
   ↓
5. Pre-Training Blockchain Storage
   ↓
6. Training Loop (with epoch tracking)
   ↓
7. Post-Training Blockchain Storage
   ↓
8. Verification & Reporting
```

### Blockchain Transaction Flow
```
1. Merkle Root Generation
   ↓
2. Transaction Construction
   ↓
3. Private Key Signing
   ↓
4. Network Broadcasting
   ↓
5. Confirmation Tracking
   ↓
6. Verification
```

### Verification Flow
```
1. Load Provenance Data
   ↓
2. Hash Verification
   ↓
3. Merkle Tree Validation
   ↓
4. Blockchain Verification
   ↓
5. Report Generation
```

## Integration Points

### External Dependencies
**Blockchain Networks**:
- Ethereum (via Web3.py)
- IPFS (via HTTP API)
- Bitcoin (via RPC)

**ML Frameworks**:
- PyTorch (core ML framework)
- Opacus (differential privacy)
- NumPy (numerical operations)

**Cryptographic Libraries**:
- BLAKE3 (hashing)
- SHA256/SHA512 (hashing)
- Web3.py (blockchain)

### Configuration Management
**Configuration Sources**:
- `configs/blockchain_config.json`: Blockchain settings
- `configs/training_config_*.json`: Training configurations
- Environment variables: Security credentials
- Command-line arguments: Runtime options

### Logging and Monitoring
**Logging Levels**:
- DEBUG: Detailed debugging information
- INFO: General operational information
- WARNING: Warning messages
- ERROR: Error conditions
- CRITICAL: Critical system failures

**Monitoring Points**:
- Training progress
- Blockchain transactions
- Hash generation
- Verification results

## Deployment Architecture

### Development Environment
**Requirements**:
- Python 3.8+
- Virtual environment
- Local Geth node
- IPFS daemon (optional)

**Setup**:
```bash
# Environment setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Blockchain setup
geth --dev --http --http.api personal,eth,net,web3,miner
```

### Production Environment
**Requirements**:
- Container orchestration (Docker/Kubernetes)
- Production blockchain nodes
- Monitoring and alerting
- Backup and recovery

**Security Considerations**:
- Secure key management
- Network isolation
- Access control
- Audit logging

## Performance Considerations

### Hash Generation Performance
**Optimizations**:
- Parallel hash computation
- Caching of intermediate results
- Algorithm selection based on requirements
- Batch processing

### Blockchain Performance
**Optimizations**:
- Gas price optimization
- Transaction batching
- Network selection
- Fallback mechanisms

### Training Performance
**Optimizations**:
- GPU acceleration
- Batch size optimization
- Memory management
- Parallel processing

## Monitoring and Logging

### Logging Strategy
**Log Categories**:
- Training logs
- Blockchain logs
- Verification logs
- Error logs

**Log Storage**:
- File-based logging
- Structured logging (JSON)
- Log rotation
- Centralized logging (optional)

### Monitoring Metrics
**Key Metrics**:
- Training accuracy and loss
- Blockchain transaction success rate
- Hash generation time
- Verification success rate

**Alerting**:
- Training failures
- Blockchain transaction failures
- Verification failures
- System errors

## Testing Strategy

### Unit Testing
**Test Coverage**:
- Hash generation functions
- Blockchain interfaces
- Provenance tracking
- Verification logic

### Integration Testing
**Test Scenarios**:
- End-to-end training pipeline
- Blockchain transaction flow
- Verification process
- Error handling

### Performance Testing
**Test Types**:
- Hash generation performance
- Blockchain transaction performance
- Training performance
- Memory usage

### Security Testing
**Test Areas**:
- Private key handling
- Hash algorithm security
- Blockchain transaction security
- Data integrity verification

## Future Enhancements

### Planned Features
1. **Multi-Blockchain Support**: Additional blockchain networks
2. **Advanced Privacy**: Enhanced differential privacy features
3. **Distributed Training**: Multi-node training support
4. **Real-time Monitoring**: Live training and blockchain monitoring
5. **API Integration**: RESTful API for external access

### Scalability Improvements
1. **Horizontal Scaling**: Multi-instance deployment
2. **Database Integration**: Persistent storage for large datasets
3. **Caching Layer**: Redis-based caching
4. **Load Balancing**: Traffic distribution

### Security Enhancements
1. **Hardware Security Modules**: HSM integration
2. **Zero-Knowledge Proofs**: Privacy-preserving verification
3. **Multi-Signature Support**: Enhanced transaction security
4. **Audit Trail**: Comprehensive audit logging

---

*This document provides a comprehensive overview of the MNIST Provenance Project architecture and design. For implementation details, refer to the individual component documentation and source code.* 