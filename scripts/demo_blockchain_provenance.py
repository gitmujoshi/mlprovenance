#!/usr/bin/env python3
"""
Demonstration script for blockchain-based ML provenance tracking.

This script shows how to:
1. Initialize blockchain-enabled provenance tracking
2. Store Merkle tree hashes before and after training
3. Verify provenance on blockchain
4. Generate blockchain reports
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_provenance.provenance.tracker import ProvenanceTracker
from ml_provenance.provenance.blockchain import ProvenanceBlockchainTracker

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_blockchain_config(config_path: str = "configs/blockchain_config.json") -> dict:
    """Load blockchain configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Blockchain config file not found at {config_path}")
        print("Using default configuration...")
        return {
            "blockchain": {
                "enabled": True,
                "networks": ["ipfs"],
                "ipfs": {
                    "enabled": True,
                    "url": "http://localhost:5001"
                }
            }
        }

def create_mock_training_data():
    """Create mock training data for demonstration."""
    import numpy as np
    
    # Mock training configuration
    training_config = {
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "model_type": "CNN",
        "dataset": "MNIST",
        "timestamp": datetime.now().isoformat()
    }
    
    # Mock training results
    training_results = {
        "final_accuracy": 0.95,
        "final_loss": 0.15,
        "training_time": 120.5,
        "epochs_completed": 3,
        "privacy_metrics": {
            "epsilon": 1.0,
            "delta": 1e-5
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Mock data statistics
    data_stats = {
        "train_samples": 60000,
        "test_samples": 10000,
        "input_shape": [28, 28, 1],
        "num_classes": 10
    }
    
    return training_config, training_results, data_stats

def demonstrate_blockchain_provenance():
    """Demonstrate blockchain provenance tracking."""
    logger = setup_logging()
    
    # Load blockchain configuration
    config = load_blockchain_config()
    
    logger.info("=== Blockchain ML Provenance Demonstration ===")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize provenance tracker with blockchain support
    logger.info("Initializing provenance tracker with blockchain support...")
    provenance_tracker = ProvenanceTracker(config=config)
    
    # Create mock data
    training_config, training_results, data_stats = create_mock_training_data()
    
    # Simulate data tracking
    logger.info("Simulating data provenance tracking...")
    import numpy as np
    mock_train_data = np.random.rand(1000, 28, 28)
    mock_test_data = np.random.rand(200, 28, 28)
    provenance_tracker.track_data(mock_train_data, mock_test_data)
    
    # Simulate model tracking
    logger.info("Simulating model provenance tracking...")
    import torch.nn as nn
    mock_model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    provenance_tracker.track_model(mock_model)
    
    # Store Merkle root hash on blockchain before training
    logger.info("=== Storing Pre-Training Merkle Root Hash ===")
    before_transactions = provenance_tracker.store_merkle_on_blockchain_before_training(training_config)
    
    if before_transactions:
        logger.info("✅ Pre-training hash stored successfully!")
        for network, tx_id in before_transactions.items():
            logger.info(f"  {network}: {tx_id}")
    else:
        logger.warning("❌ Failed to store pre-training hash")
    
    # Simulate training completion
    logger.info("=== Simulating Training Completion ===")
    logger.info("Training completed successfully!")
    
    # Store Merkle root hash on blockchain after training
    logger.info("=== Storing Post-Training Merkle Root Hash ===")
    after_transactions = provenance_tracker.store_merkle_on_blockchain_after_training(training_results)
    
    if after_transactions:
        logger.info("✅ Post-training hash stored successfully!")
        for network, tx_id in after_transactions.items():
            logger.info(f"  {network}: {tx_id}")
    else:
        logger.warning("❌ Failed to store post-training hash")
    
    # Verify blockchain provenance
    logger.info("=== Verifying Blockchain Provenance ===")
    verification_results = provenance_tracker.verify_blockchain_provenance()
    
    logger.info("Verification Results:")
    logger.info(json.dumps(verification_results, indent=2))
    
    if verification_results.get("chain_integrity", False):
        logger.info("✅ Blockchain provenance verification successful!")
    else:
        logger.warning("❌ Blockchain provenance verification failed!")
    
    # Get blockchain status
    logger.info("=== Blockchain Status ===")
    blockchain_status = provenance_tracker.get_blockchain_status()
    logger.info(json.dumps(blockchain_status, indent=2))
    
    # Save blockchain report
    logger.info("=== Saving Blockchain Report ===")
    report_path = provenance_tracker.save_blockchain_report()
    if report_path:
        logger.info(f"✅ Blockchain report saved to: {report_path}")
    else:
        logger.warning("❌ Failed to save blockchain report")
    
    # Save provenance data
    logger.info("=== Saving Provenance Data ===")
    provenance_path = provenance_tracker.save()
    logger.info(f"✅ Provenance data saved to: {provenance_path}")
    
    logger.info("=== Demonstration Complete ===")
    
    return {
        "before_transactions": before_transactions,
        "after_transactions": after_transactions,
        "verification_results": verification_results,
        "blockchain_status": blockchain_status,
        "report_path": report_path,
        "provenance_path": provenance_path
    }

def demonstrate_blockchain_interfaces():
    """Demonstrate different blockchain interfaces."""
    logger = setup_logging()
    
    logger.info("=== Blockchain Interface Demonstration ===")
    
    # Test IPFS interface
    logger.info("Testing IPFS interface...")
    from ml_provenance.provenance.blockchain import IPFSInterface
    
    ipfs_interface = IPFSInterface("http://localhost:5001")
    
    test_data = {
        "merkle_root_hash": "test_hash_123",
        "timestamp": datetime.now().isoformat(),
        "type": "test"
    }
    
    try:
        cid = ipfs_interface.store_hash("test_hash_123", test_data)
        logger.info(f"✅ IPFS storage successful: {cid}")
        
        is_valid = ipfs_interface.verify_hash("test_hash_123", cid)
        logger.info(f"✅ IPFS verification: {'SUCCESS' if is_valid else 'FAILED'}")
        
        tx_info = ipfs_interface.get_transaction_info(cid)
        logger.info(f"✅ IPFS transaction info: {tx_info}")
        
    except Exception as e:
        logger.warning(f"❌ IPFS test failed: {e}")
    
    # Test Ethereum interface (simulation)
    logger.info("Testing Ethereum interface (simulation)...")
    from ml_provenance.provenance.blockchain import EthereumInterface
    
    try:
        eth_interface = EthereumInterface("http://localhost:8545")
        tx_id = eth_interface.store_hash("test_hash_456", test_data)
        logger.info(f"✅ Ethereum storage simulation: {tx_id}")
        
        is_valid = eth_interface.verify_hash("test_hash_456", tx_id)
        logger.info(f"✅ Ethereum verification simulation: {'SUCCESS' if is_valid else 'FAILED'}")
        
    except Exception as e:
        logger.warning(f"❌ Ethereum test failed: {e}")
    
    # Test Bitcoin interface (simulation)
    logger.info("Testing Bitcoin interface (simulation)...")
    from ml_provenance.provenance.blockchain import BitcoinInterface
    
    try:
        btc_interface = BitcoinInterface("http://localhost:8332")
        tx_id = btc_interface.store_hash("test_hash_789", test_data)
        logger.info(f"✅ Bitcoin storage simulation: {tx_id}")
        
        is_valid = btc_interface.verify_hash("test_hash_789", tx_id)
        logger.info(f"✅ Bitcoin verification simulation: {'SUCCESS' if is_valid else 'FAILED'}")
        
    except Exception as e:
        logger.warning(f"❌ Bitcoin test failed: {e}")

def main():
    """Main demonstration function."""
    print("🚀 Starting Blockchain ML Provenance Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate blockchain interfaces
        demonstrate_blockchain_interfaces()
        print()
        
        # Demonstrate full provenance tracking
        results = demonstrate_blockchain_provenance()
        
        print("\n" + "=" * 60)
        print("📊 Demonstration Summary:")
        print(f"  • Pre-training transactions: {len(results['before_transactions'] or {})}")
        print(f"  • Post-training transactions: {len(results['after_transactions'] or {})}")
        print(f"  • Chain integrity: {'✅' if results['verification_results'].get('chain_integrity') else '❌'}")
        print(f"  • Blockchain enabled: {'✅' if results['blockchain_status'].get('blockchain_enabled') else '❌'}")
        print(f"  • Report saved: {'✅' if results['report_path'] else '❌'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 