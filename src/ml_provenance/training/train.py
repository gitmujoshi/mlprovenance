#!/usr/bin/env python3
"""
MNIST Training with Provenance Tracking and Blockchain Integration

This module implements a complete machine learning training pipeline with:
- MNIST dataset training using PyTorch
- Differential privacy with Opacus
- Comprehensive provenance tracking
- Blockchain integration for immutable storage
- Merkle tree verification
- Multi-hash algorithm support

Architecture:
├── Data Loading & Preprocessing
├── Model Creation & Architecture Tracking
├── Provenance Initialization
├── Pre-Training Blockchain Storage
├── Training Loop with Epoch Tracking
├── Post-Training Blockchain Storage
└── Verification & Reporting

Key Components:
- ProvenanceTracker: Manages complete audit trail
- BlockchainManager: Handles multi-network blockchain operations
- HashFactory: Configurable cryptographic hashing
- MerkleTree: Cryptographic proof of data integrity
- VerificationEngine: Validates provenance integrity

Security Features:
- Environment-based private key management
- Configurable hash algorithms (BLAKE3, SHA256, SHA512)
- Differential privacy with configurable epsilon
- Blockchain immutability for provenance hashes
- Comprehensive audit logging

Author: ML Provenance Team
License: MIT
"""

import os
import urllib.request
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
import sys
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Tuple, Dict, Any
import torch.nn.functional as F
import io
import contextlib
from datetime import datetime
import json
import time
import argparse

# Update imports to use package imports
from ml_provenance.models.mnist_model import MNISTModel
from ml_provenance.data.mnist_data import get_mnist_data
from ml_provenance.provenance.tracker import Tracker, ProvenanceTracker
from ml_provenance.provenance.verifier import Verifier, ProvenanceVerifier
from ml_provenance.provenance.report_generator import ReportGenerator
from ml_provenance.provenance.generate_final_report import generate_final_report

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# URLs for MNIST data
MNIST_MIRRORS = {
    'train_images': [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz'
    ],
    'train_labels': [
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz'
    ],
    'test_images': [
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz'
    ],
    'test_labels': [
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'
    ]
}

DATA_DIR = project_root / 'data'


class ConsoleCapture:
    """Capture console output and save to file."""
    
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.log_buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        # Redirect stdout and stderr to our buffer
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Write captured output to file
        with open(self.log_file_path, 'w') as f:
            f.write(self.log_buffer.getvalue())
        
        self.log_buffer.close()
    
    def write(self, text):
        # Write to both original stdout and our buffer
        self.original_stdout.write(text)
        self.log_buffer.write(text)
        self.original_stdout.flush()
    
    def flush(self):
        self.original_stdout.flush()


def setup_logging(provenance_dir: Path) -> Tuple[logging.Logger, Path]:
    """Set up comprehensive logging to both console and file."""
    
    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = provenance_dir / f"training_run_log_{timestamp}.txt"
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )
    
    logger = logging.getLogger(__name__)
    
    return logger, log_file_path


def download_mnist():
    DATA_DIR.mkdir(exist_ok=True)
    for key, urls in MNIST_MIRRORS.items():
        out_path = DATA_DIR / urls[0].split('/')[-1]
        if not out_path.exists():
            print(f"Downloading {key}...")
            for url in urls:
                try:
                    print(f"Trying {url}...")
                    urllib.request.urlretrieve(url, out_path)
                    print(f"Successfully downloaded from {url}")
                    break
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
            else:
                raise Exception(f"Could not download {key} from any mirror")


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows, cols) / 255.0
        return data


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def get_mnist_datasets():
    download_mnist()
    train_images = load_mnist_images(DATA_DIR / 'train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels(DATA_DIR / 'train-labels-idx1-ubyte.gz')
    test_images = load_mnist_images(DATA_DIR / 't10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels(DATA_DIR / 't10k-labels-idx1-ubyte.gz')
    return train_images, train_labels, test_images, test_labels

def create_model():
    """Create the MNIST model."""
    model = MNISTModel()
    return model

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    privacy_engine: PrivacyEngine,
    epochs: int,
    device: str,
    provenance_tracker: ProvenanceTracker,
    target_delta: float,
    noise_multiplier: float,
    config: dict
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the model with differential privacy."""
    model = model.to(device)
    model.train()
    
    # Initialize tracking
    training_history = []
    best_accuracy = 0.0
    last_epoch_loss = 0.0
    last_epoch_train_acc = 0.0
    last_epoch_test_acc = 0.0
    privacy_budget = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total  # fraction
        
        # Test the model
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_accuracy = test_correct / test_total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # Get privacy metrics from the privacy engine
        current_epsilon = privacy_engine.get_epsilon(target_delta)
        privacy_budget.append(current_epsilon)
        privacy_metrics = {
            'epsilon': current_epsilon,
            'delta': target_delta,
            'noise_multiplier': noise_multiplier,
            'privacy_budget': privacy_budget
        }
        
        # Track epoch metrics and model state
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': epoch_loss,  # Using same loss for now
            'train_acc': epoch_accuracy,
            'val_acc': test_accuracy,
            'model_state': model.state_dict(),  # Capture model state
            'is_final': epoch == epochs - 1,
            'privacy_metrics': privacy_metrics
        }
        
        # Add to training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': epoch_loss,
            'train_acc': epoch_accuracy,
            'val_acc': test_accuracy
        })
        
        # Update provenance with epoch data
        provenance_tracker.update_training_provenance(epoch_data)
        
        last_epoch_loss = epoch_loss
        last_epoch_train_acc = epoch_accuracy
        last_epoch_test_acc = test_accuracy
    
    final_metrics = {
        'loss': last_epoch_loss,
        'train_accuracy': last_epoch_train_acc,
        'test_accuracy': last_epoch_test_acc,
        'training_history': training_history,
        'privacy_metrics': {
            'target_epsilon': config['privacy_parameters']['target_epsilon'],
            'final_epsilon': privacy_budget[-1] if privacy_budget else 0,
            'delta': target_delta,
            'privacy_budget': privacy_budget
        }
    }
    
    return model, final_metrics

def verify_blockchain_report(provenance_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Comprehensive blockchain verification function."""
    verification_results = {
        "blockchain_enabled": False,
        "has_transactions": False,
        "chain_integrity": False,
        "ethereum_connection": False,
        "private_key_set": False,
        "verification_status": "failed"
    }
    
    try:
        blockchain_report_path = provenance_dir / "blockchain_report.json"
        if not blockchain_report_path.exists():
            logger.warning("Blockchain report not found")
            return verification_results
        
        with open(blockchain_report_path, 'r') as f:
            blockchain_report = json.load(f)
        
        logger.info("=" * 60)
        logger.info("🔍 BLOCKCHAIN VERIFICATION REPORT")
        logger.info("=" * 60)
        
        blockchain_config = blockchain_report.get("blockchain_config", {})
        verification_results["blockchain_enabled"] = blockchain_config.get("enabled", False)
        
        eth_config = blockchain_config.get("ethereum", {})
        verification_results["ethereum_enabled"] = eth_config.get("enabled", False)
        
        # Check for private key in config or environment variable
        config_private_key = eth_config.get("private_key")
        env_private_key = os.getenv('ETH_PRIVATE_KEY')
        verification_results["private_key_set"] = (config_private_key is not None and config_private_key != "null") or env_private_key is not None
        
        stored_hashes = blockchain_report.get("stored_hashes", {})
        has_transactions = any(bool(data.get('transaction_ids', {})) for data in stored_hashes.values())
        verification_results["has_transactions"] = has_transactions
        
        verification_data = blockchain_report.get("verification_results", {})
        verification_results["chain_integrity"] = verification_data.get("chain_integrity", False)
        
        # Test Ethereum connection
        try:
            import requests
            response = requests.post("http://127.0.0.1:8545", json={"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}, timeout=5)
            verification_results["ethereum_connection"] = response.status_code == 200
        except Exception as e:
            logger.warning(f"Ethereum connection test failed: {e}")
            verification_results["ethereum_connection"] = False
        
        # Log results
        logger.info(f"📊 Blockchain Configuration:")
        logger.info(f"  Enabled: {'✅' if verification_results['blockchain_enabled'] else '❌'}")
        logger.info(f"  Ethereum Enabled: {'✅' if verification_results['ethereum_enabled'] else '❌'}")
        logger.info(f"  Private Key Set: {'✅' if verification_results['private_key_set'] else '❌'}")
        logger.info(f"  Ethereum Connection: {'✅' if verification_results['ethereum_connection'] else '❌'}")
        
        logger.info(f"\n📊 Stored Hashes:")
        for stage, data in stored_hashes.items():
            logger.info(f"  {stage}:")
            logger.info(f"    Merkle Root: {data.get('merkle_root_hash', 'N/A')[:16]}...")
            tx_ids = data.get('transaction_ids', {})
            logger.info(f"    Transaction IDs: {tx_ids}")
            if tx_ids:
                logger.info(f"    ✅ Transactions found")
            else:
                logger.info(f"    ❌ No transactions found")
        
        logger.info(f"\n🔍 Verification Results:")
        logger.info(f"  Chain Integrity: {'✅' if verification_results['chain_integrity'] else '❌'}")
        logger.info(f"  Has Transactions: {'✅' if verification_results['has_transactions'] else '❌'}")
        
        # Determine status
        if (verification_results["blockchain_enabled"] and verification_results["ethereum_enabled"] and 
            verification_results["ethereum_connection"] and verification_results["has_transactions"] and 
            verification_results["chain_integrity"]):
            verification_results["verification_status"] = "success"
            logger.info(f"\n🎉 BLOCKCHAIN VERIFICATION: SUCCESS ✅")
        elif verification_results["blockchain_enabled"] and verification_results["ethereum_connection"]:
            verification_results["verification_status"] = "partial"
            logger.info(f"\n⚠️  BLOCKCHAIN VERIFICATION: PARTIAL ⚠️")
        else:
            verification_results["verification_status"] = "failed"
            logger.info(f"\n❌ BLOCKCHAIN VERIFICATION: FAILED ❌")
        
        logger.info("=" * 60)
        return verification_results
        
    except Exception as e:
        logger.error(f"Error during blockchain verification: {e}")
        verification_results["verification_status"] = "error"
        return verification_results

def main():
    """
    Main entry point for the MNIST training pipeline with provenance tracking.
    
    This function orchestrates the complete training pipeline including:
    1. Configuration loading and validation
    2. Data loading and preprocessing
    3. Model initialization
    4. Provenance tracker setup
    5. Pre-training blockchain storage
    6. Training execution with epoch tracking
    7. Post-training blockchain storage
    8. Verification and reporting
    
    The entire process is logged and tracked for auditability, with
    all artifacts being stored on the blockchain for immutability.
    
    Environment Variables:
        ETH_PRIVATE_KEY: Ethereum private key for blockchain transactions
        
    Configuration Files:
        configs/training_config_*.json: Training hyperparameters
        configs/blockchain_config.json: Blockchain integration settings
        
    Output:
        - Trained model artifacts
        - Provenance reports
        - Blockchain transaction records
        - Verification reports
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST with provenance tracking')
    parser.add_argument('--config', type=str, default='configs/training_config_blake3.json',
                       help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='artifacts',
                       help='Directory to save training artifacts')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Capture console output for audit purposes
    with ConsoleCapture(output_dir / "console_output.txt"):
        logger, _ = setup_logging(output_dir)
        logger.info("Starting MNIST training with provenance tracking")
        
        # Load training configuration
        config_path = Path(__file__).parent.parent.parent.parent / args.config
        config = load_training_config(str(config_path))
        
        # Load blockchain configuration from external file
        blockchain_config_path = Path(__file__).parent.parent.parent.parent / "configs" / "blockchain_config.json"
        
        if blockchain_config_path.exists():
            with open(blockchain_config_path, 'r') as f:
                blockchain_config = json.load(f)
            # logger.info(f"Loaded blockchain configuration from: {blockchain_config_path}")
            print(f"Loaded blockchain configuration from: {blockchain_config_path}")
        else:
            # logger.warning(f"Blockchain config not found at {blockchain_config_path}, using default configuration")
            print(f"Blockchain config not found at {blockchain_config_path}, using default configuration")
            blockchain_config = {
                "blockchain": {
                    "enabled": True,
                    "networks": ["ipfs"],
                    "ipfs": {
                        "enabled": True,
                        "url": "http://localhost:5001",
                        "timeout": 30,
                        "retry_attempts": 3
                    },
                    "ethereum": {
                        "enabled": False,
                        "rpc_url": "http://127.0.0.1:8545",
                        "private_key": None,
                        "contract_address": None,
                        "gas_limit": 300000,
                        "gas_price": "auto"
                    }
                }
            }
        
        # Merge blockchain config into main config
        config['blockchain'] = blockchain_config.get('blockchain', {})
        
        # Initialize hash configuration for provenance tracking
        hash_config = TrainingHashConfig(
            algorithm=config.get('hash_algorithm', 'blake3'),
            include_timestamps=True,
            include_metadata=True
        )
        
        # Initialize provenance tracker with configuration
        provenance_tracker = ProvenanceTracker(config=config)
        logger.info("Provenance tracker initialized")
        
        # Load MNIST dataset
        train_data, test_data = get_mnist_datasets()
        logger.info(f"Loaded MNIST dataset: {len(train_data)} training samples, {len(test_data)} test samples")
        
        # Track data provenance
        provenance_tracker.track_data(train_data, test_data)
        logger.info("Data provenance tracked")
        
        # Setup data loaders
        training_config = config.get('training', {})
        batch_size = training_config.get('batch_size', 64)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Initialize model and optimizer
        model = create_model().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Track model provenance
        provenance_tracker.track_model(model)
        logger.info("Model provenance tracked")
        
        # Pre-training blockchain storage
        logger.info("Storing pre-training provenance on blockchain...")
        pre_training_provenance = provenance_tracker.get_provenance_summary()
        blockchain_report = provenance_tracker.store_provenance_on_blockchain(pre_training_provenance)
        
        # Initialize privacy engine
        logger.info("Initializing privacy engine...")
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=config["privacy_parameters"]["noise_multiplier"],
            max_grad_norm=config["privacy_parameters"]["max_grad_norm"],
        )
        
        # Execute training
        logger.info("Starting model training...")
        logger.info("-" * 50)
        model, final_metrics = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            privacy_engine=privacy_engine,
            epochs=config["epochs"],
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            provenance_tracker=provenance_tracker,
            target_delta=config["privacy_parameters"]["target_delta"],
            noise_multiplier=config["privacy_parameters"]["noise_multiplier"],
            config=config
        )
        logger.info("-" * 50)
        logger.info("Training completed!")
        
        # Track final training results
        provenance_tracker.track_training(model, config, final_metrics)
        logger.info("Training provenance tracked")
        
        # Post-training blockchain storage
        logger.info("Storing post-training provenance on blockchain...")
        post_training_provenance = provenance_tracker.get_provenance_summary()
        blockchain_report = provenance_tracker.store_provenance_on_blockchain(post_training_provenance)
        
        # Save model
        model_path = output_dir / "mnist_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Generate final provenance report
        final_provenance = provenance_tracker.get_provenance_summary()
        provenance_path = output_dir / "provenance_report.json"
        with open(provenance_path, 'w') as f:
            json.dump(final_provenance, f, indent=2, default=str)
        logger.info(f"Provenance report saved to: {provenance_path}")
        
        # Verify blockchain integration
        verification_results = verify_blockchain_report(output_dir, logger)
        
        # Generate verification report
        verification_path = output_dir / "verification_report.json"
        with open(verification_path, 'w') as f:
            json.dump(verification_results, f, indent=2, default=str)
        logger.info(f"Verification report saved to: {verification_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Final Test Accuracy: {final_metrics['test_accuracy']:.2f}%")
        print(f"Blockchain Verification: {verification_results['verification_status'].upper()}")
        print(f"Model Saved: {model_path}")
        print(f"Provenance Report: {provenance_path}")
        print(f"Verification Report: {verification_path}")
        print("="*60)

if __name__ == "__main__":
    main() 