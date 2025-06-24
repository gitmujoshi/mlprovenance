# MNIST Provenance Tracking with Blockchain Integration

A comprehensive machine learning provenance tracking system with blockchain integration for immutable, tamper-evident audit trails.

## 🚀 Features

- **🔗 Multi-Blockchain Support**: IPFS, Ethereum, Bitcoin
- **📊 Merkle Tree Integration**: Cryptographic verification of ML pipeline
- **🔒 Immutable Provenance**: Tamper-evident audit trails
- **⚡ Auto Mode**: Fully automated blockchain integration
- **🛠️ Developer Friendly**: Easy setup and configuration
- **🔐 Privacy-Preserving**: Differential privacy support
- **📈 Comprehensive Tracking**: Data, model, and training provenance

## 🏗️ Architecture

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

## 🚀 Quick Start

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

## 📚 Documentation

- **[Developer Guide](docs/developer_guide.md)** - Comprehensive guide for developers
- **[Blockchain Integration](docs/blockchain_provenance.md)** - Detailed blockchain documentation
- **[Architecture](ARCHITECTURE.md)** - System architecture overview
- **[Provenance Tracking](docs/provenance_tracking.md)** - Provenance tracking concepts
- **[Model Training](docs/model_training_and_safety.md)** - Training and safety features

## 🔧 Configuration

### Blockchain Configuration

```json
{
  "blockchain": {
    "enabled": true,
    "networks": ["ipfs", "ethereum"],
    "ipfs": {
      "enabled": true,
      "url": "http://localhost:5001"
    },
    "ethereum": {
      "enabled": true,
      "rpc_url": "http://127.0.0.1:8545",
      "private_key": null
    }
  }
}
```

**⚠️ Security Best Practices:**

1. **Never store private keys in configuration files**
2. **Use environment variables for sensitive data:**
   ```bash
   # Set environment variables
   export ETH_PRIVATE_KEY=your_ethereum_private_key_here
   export BTC_PRIVATE_KEY=your_bitcoin_private_key_here
   
   # Or use a .env file (copy from env.example)
   cp env.example .env
   # Edit .env with your actual private keys
   ```

3. **Add .env to .gitignore to prevent accidental commits**
4. **Use testnet keys for development**
5. **Rotate keys regularly in production**

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
            "private_key": None  # Will use ETH_PRIVATE_KEY environment variable
        }
    }
}
```

## 💻 Usage Examples

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
before_transactions = provenance_tracker.store_merkle_on_blockchain_before_training(training_config)

# ... training process ...

# Store post-training hash on blockchain
after_transactions = provenance_tracker.store_merkle_on_blockchain_after_training(training_results)

# Verify blockchain provenance
verification_results = provenance_tracker.verify_blockchain_provenance()
```

### Advanced Usage

```python
# Get blockchain status
status = provenance_tracker.get_blockchain_status()
print(f"Blockchain enabled: {status['blockchain_enabled']}")

# Verify provenance chain
verification = provenance_tracker.verify_blockchain_provenance()
if verification['chain_integrity']:
    print("✅ Provenance chain integrity verified!")
```

## 🔗 Supported Blockchain Networks

### IPFS (InterPlanetary File System)
- **Advantages**: Decentralized storage, no fees, high availability
- **Setup**: `brew install ipfs && ipfs daemon`
- **Use Case**: Development and testing

### Ethereum
- **Advantages**: Smart contracts, immutable blockchain, programmable verification
- **Setup**: `brew install ethereum && bash scripts/setup_local_geth.sh`
- **Use Case**: Production environments

### Bitcoin
- **Advantages**: Maximum security, global consensus, long-term stability
- **Setup**: `brew install bitcoin && bitcoind`
- **Use Case**: High-security requirements

## 🛠️ Development

### Prerequisites

- Python 3.8+
- Git
- Homebrew (for macOS)

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd mnist_provenance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8

# Run tests
python -m pytest tests/
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/

# Integration tests
python scripts/demo_blockchain_provenance.py

# Full training test
python src/ml_provenance/training/train.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: `pip install gitpython web3 requests ipfshttpclient`
2. **Geth Connection**: Check if Geth is running with `lsof -i :8545`
3. **IPFS Connection**: Start IPFS daemon with `ipfs daemon`
4. **Private Key Issues**: Extract from Geth dev node keystore

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configuration
config["blockchain"]["debug"] = True
```

## 📊 Output Files

The system generates several output files:

- `blockchain_report.json` - Complete blockchain verification report
- `provenance_report.json` - Standard provenance report with blockchain info
- `merkle_tree_*.json` - Merkle tree structure files
- `geth_dev.log` - Geth development node logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:

1. Check the [troubleshooting section](docs/developer_guide.md#troubleshooting)
2. Review the [API documentation](docs/developer_guide.md#api-reference)
3. Check existing issues on GitHub
4. Create a new issue with detailed information

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Web3.py](https://web3py.readthedocs.io/) for Ethereum integration
- [IPFS](https://ipfs.io/) for decentralized storage
- [Opacus](https://opacus.ai/) for differential privacy

---

**Made with ❤️ for secure and verifiable machine learning** 