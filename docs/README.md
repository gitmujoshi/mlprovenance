# Documentation Overview: ML Provenance Tracking with Blockchain Integration

Welcome to the comprehensive documentation for the ML provenance tracking system with blockchain integration. This system provides immutable, tamper-evident audit trails for machine learning training processes.

## 📚 Documentation Structure

### 🚀 Getting Started
- **[README.md](../README.md)** - Project overview and quick start guide
- **[Developer Guide](developer_guide.md)** - Comprehensive guide for developers
- **[User Documentation](user_documentation.md)** - User-friendly guide for end users

### 🏗️ Architecture & Design
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture overview
- **[Blockchain Integration](blockchain_provenance.md)** - Detailed blockchain documentation
- **[Provenance Tracking](provenance_tracking.md)** - Provenance tracking concepts

### 📖 Technical Documentation
- **[Comprehensive Guide](comprehensive_guide.md)** - Complete system guide
- **[Model Training](model_training_and_safety.md)** - Training and safety features
- **[Technical Documentation](technical_documentation.md)** - Technical implementation details

### 📄 Reports & Examples
- **[Training Run Report](training_run_report.md)** - Example training report
- **[Academic Paper](academic_paper.md)** - Research paper on the system
- **[Provenance](provenance.md)** - Provenance tracking documentation

## 🎯 Quick Navigation

### For New Users
1. Start with **[README.md](../README.md)** for project overview
2. Follow **[User Documentation](user_documentation.md)** for step-by-step usage
3. Run the demo: `python3 scripts/demo_blockchain_provenance.py`

### For Developers
1. Read **[Developer Guide](developer_guide.md)** for comprehensive development information
2. Review **[ARCHITECTURE.md](../ARCHITECTURE.md)** for system design
3. Check **[Blockchain Integration](blockchain_provenance.md)** for blockchain features

### For System Administrators
1. Review **[ARCHITECTURE.md](../ARCHITECTURE.md)** for deployment considerations
2. Check **[Technical Documentation](technical_documentation.md)** for implementation details
3. Follow **[Comprehensive Guide](comprehensive_guide.md)** for complete system understanding

## 🔗 Key Features

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

## 📋 Configuration

### Blockchain Configuration
Create `configs/blockchain_config.json`:
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
      "private_key": "your_private_key_here"
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

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: `pip install gitpython`
2. **Geth Connection**: Check if Geth is running with `lsof -i :8545`
3. **IPFS Connection**: Start IPFS daemon with `ipfs daemon`
4. **Private Key Issues**: Extract from Geth dev node keystore
5. **Gas Limit Issues**: Increase gas limit in configuration

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

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting sections in the documentation
2. Review the [developer guide](developer_guide.md)
3. Check existing issues on GitHub
4. Create a new issue with detailed information

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Web3.py](https://web3py.readthedocs.io/) for Ethereum integration
- [IPFS](https://ipfs.io/) for decentralized storage
- [Opacus](https://opacus.ai/) for differential privacy

---

## 📖 Documentation Index

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](../README.md) | Project overview and quick start | All users |
| [Developer Guide](developer_guide.md) | Comprehensive development guide | Developers |
| [User Documentation](user_documentation.md) | User-friendly usage guide | End users |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | System architecture overview | Developers, Architects |
| [Blockchain Integration](blockchain_provenance.md) | Blockchain features documentation | Developers |
| [Comprehensive Guide](comprehensive_guide.md) | Complete system guide | All users |
| [Provenance Tracking](provenance_tracking.md) | Provenance concepts | Users, Developers |
| [Model Training](model_training_and_safety.md) | Training and safety features | Users, Developers |
| [Technical Documentation](technical_documentation.md) | Technical implementation details | Developers |
| [Training Run Report](training_run_report.md) | Example training report | Users, Analysts |
| [Academic Paper](academic_paper.md) | Research paper | Researchers, Academics |
| [Provenance](provenance.md) | Provenance tracking documentation | Users, Developers |

---

**Made with ❤️ for secure and verifiable machine learning** 