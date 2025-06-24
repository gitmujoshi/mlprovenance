"""
Provenance Tracking System for Machine Learning

This module implements a comprehensive provenance tracking system for ML workflows,
providing complete audit trails of data, models, and training processes. It integrates
with blockchain networks for immutable storage and verification.

Architecture:
├── ProvenanceTracker: Main tracking coordinator
├── Data Provenance: Dataset tracking and statistics
├── Model Provenance: Architecture and parameter tracking
├── Training Provenance: Process and metrics tracking
├── Blockchain Integration: Immutable storage
└── Verification: Integrity checking and validation

Key Features:
- Complete audit trail of ML lifecycle
- Multi-hash algorithm support (BLAKE3, SHA256, SHA512)
- Merkle tree construction for data integrity
- Blockchain integration for immutability
- Differential privacy tracking
- Comprehensive metadata collection
- Cross-platform compatibility

Data Flow:
1. Data Loading → Statistics Calculation → Hash Generation
2. Model Creation → Architecture Extraction → Parameter Counting
3. Training Process → Metrics Collection → Epoch Tracking
4. Blockchain Storage → Merkle Root → Transaction Recording
5. Verification → Integrity Check → Report Generation

Security Features:
- Cryptographic hash verification
- Blockchain immutability
- Audit logging
- Data integrity checks
- Privacy-preserving metrics

Author: ML Provenance Team
License: MIT
"""

import json
import os
import sys
import platform
import git
from datetime import datetime
from pathlib import Path
import logging
import torch
import numpy as np
import hashlib
import pickle
import base64
from .merkle_tree import MLProvenanceMerkleTree
from .blockchain import ProvenanceBlockchainTracker
from typing import Dict, Any, List, Optional
import opacus
import torch.nn as nn
from .provenance_data import ProvenanceData

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class Tracker:
    """Tracker for ML provenance using consistent data structure."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.provenance = ProvenanceData()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.provenance_dir = os.path.join(output_dir, "provenance", self.timestamp)
        os.makedirs(self.provenance_dir, exist_ok=True)
        
        # Set up file logging for provenance
        log_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "provenance.log")
        self.logger = logging.getLogger(f"provenance_{self.timestamp}")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(fh)
    
    def track_data(self, train_data: torch.Tensor, test_data: torch.Tensor, dataset_name: str) -> None:
        """Track data provenance."""
        data_info = {
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat()
        }
        self.provenance.update_data(data_info)
        
        # Calculate basic statistics
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        train_min = np.min(train_data)
        train_max = np.max(train_data)
        
        test_mean = np.mean(test_data)
        test_std = np.std(test_data)
        test_min = np.min(test_data)
        test_max = np.max(test_data)
        
        # For train data
        train_stats = {
            "mean": train_mean,
            "std": train_std,
            "min": float(train_min),
            "max": float(train_max),
            "shape": list(train_data.shape),
            "dtype": str(train_data.dtype)
        }
        
        # For test data
        test_stats = {
            "mean": test_mean,
            "std": test_std,
            "min": float(test_min),
            "max": float(test_max),
            "shape": list(test_data.shape),
            "dtype": str(test_data.dtype)
        }
        
        # Generate data hashes
        train_hash = self._generate_hash(train_data)
        test_hash = self._generate_hash(test_data)
        
        # Create a comprehensive data info dictionary for hash computation
        data_info = {
            "dataset": dataset_name,
            "timestamp": self.timestamp,
            "train": {
                "samples": len(train_data),
                "mean": train_mean,
                "std": train_std,
                "hash": train_hash["hash"],
                "metadata": {
                    "statistics": train_stats
                }
            },
            "test": {
                "samples": len(test_data),
                "mean": test_mean,
                "std": test_std,
                "hash": test_hash["hash"],
                "metadata": {
                    "statistics": test_stats
                }
            }
        }
        
        # Log the data_info dictionary before hashing
        self.logger.info(f"[TRACKER] data_info to be hashed: {data_info}")
        data_hash = self._generate_hash(data_info, "data")
        self.logger.info(f"[TRACKER] data hash: {data_hash['hash']}")
        
        # Keep existing structure for backward compatibility
        self.data["data_provenance"] = {
            "train": {
                "samples": len(train_data),
                "mean": train_mean,
                "std": train_std,
                "hash": train_hash["hash"],
                "metadata": {
                    "statistics": train_stats
                }
            },
            "test": {
                "samples": len(test_data),
                "mean": test_mean,
                "std": test_std,
                "hash": test_hash["hash"],
                "metadata": {
                    "statistics": test_stats
                }
            },
            "dataset": dataset_name,
            "timestamp": self.timestamp
        }
        
        # Store data hashes in the hashes section
        self.data["hashes"]["train_data"] = train_hash["hash"]
        self.data["hashes"]["test_data"] = test_hash["hash"]
        self.data["hashes"]["data"] = data_hash["hash"]
        
        self.logger.info("Data provenance tracked successfully.")
    
    def track_model(self, model: nn.Module) -> None:
        """Track model architecture (which doesn't change during training)."""
        self.logger.info("Tracking model architecture...")
        
        # Get model architecture and parameters
        model_info = {
            "architecture": {
                "name": model.__class__.__name__,
                "layers": [
                    {
                        "name": name,
                        "type": layer.__class__.__name__,
                        "parameters": sum(p.numel() for p in layer.parameters())
                    }
                    for name, layer in model.named_children()
                ],
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "timestamp": self.timestamp
            }
        }
        
        # Store architecture info
        self.data["model_provenance"] = {
            "architecture": model_info["architecture"],
            "timestamp": self.timestamp
        }
        
        self.logger.info("Model architecture tracked successfully.")
    
    def track_training(self, config: Dict[str, Any], final_metrics: Dict[str, float]) -> None:
        """Track training provenance."""
        training_info = {
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "final_metrics": final_metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.provenance.update_training(training_info)
    
    def track_privacy(self, config: Dict[str, Any], achieved_metrics: Dict[str, float]) -> None:
        """Track privacy provenance."""
        privacy_info = {
            "target_epsilon": config.get("target_epsilon"),
            "target_delta": config.get("target_delta"),
            "noise_multiplier": config.get("noise_multiplier"),
            "achieved_epsilon": achieved_metrics.get("epsilon"),
            "achieved_delta": achieved_metrics.get("delta"),
            "timestamp": datetime.now().isoformat()
        }
        self.provenance.update_privacy(privacy_info)
    
    def save(self) -> str:
        """Save provenance data to file."""
        filepath = os.path.join(self.provenance_dir, "provenance.json")
        save_data = {
            "version": self.timestamp,
            "data_provenance": self.data["data_provenance"],
            "model_provenance": self.data["model_provenance"],
            "training_provenance": self.data["training_provenance"],
            "system_info": self.data["system_info"],
            "hashes": self.data["hashes"]
        }
        serializable_data = convert_to_serializable(save_data)
        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)
        self.logger.info(f"Provenance data saved to {filepath}")
        return filepath
    
    def get_provenance_dir(self) -> str:
        """Get the provenance directory path."""
        return self.provenance_dir

class ProvenanceTracker:
    def __init__(self, base_dir="artifacts", config: Optional[Dict[str, Any]] = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir)
        self.provenance_dir = self.base_dir / "provenance" / self.timestamp
        self.provenance_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ProvenanceTracker in {self.provenance_dir}")
        
        # Initialize provenance data
        self.data = {
            "version": self.timestamp,
            "data_provenance": {},
            "model_provenance": {},
            "training_provenance": {},
            "system_info": self._get_system_info(),
            "hashes": {}
        }
        
        # Initialize Merkle tree
        self.merkle_tree = MLProvenanceMerkleTree()
        
        # Initialize blockchain tracker if config provided
        self.blockchain_tracker = None
        if config:
            self.logger.info(f"DEBUG: Config provided, keys: {list(config.keys())}")
            try:
                self.blockchain_tracker = ProvenanceBlockchainTracker(config)
                self.logger.info("DEBUG: ProvenanceBlockchainTracker created successfully")
                self.logger.info("Blockchain tracking enabled")
            except Exception as e:
                self.logger.error(f"DEBUG: Failed to create ProvenanceBlockchainTracker: {e}")
                import traceback
                self.logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
                self.blockchain_tracker = None
        else:
            self.logger.info("Blockchain tracking disabled - no config provided")
    
    def _make_serializable(self, data):
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items() if k != "model_state"}
        elif isinstance(data, list):
            return [self._make_serializable(v) for v in data]
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating, np.float32)):
            return float(data)
        else:
            return data

    def _generate_hash(self, data: Any, component_name: str = None) -> Dict[str, str]:
        """Generate a hash for the given data."""
        serializable_data = convert_to_serializable(data)
        self.logger.debug(f"Data structure for {component_name}: {json.dumps(serializable_data, indent=2)}")
        data_str = json.dumps(serializable_data, sort_keys=True)
        
        # Use the configured hash function from HashFactory
        from .hash_config import HashFactory
        hash_function = HashFactory.get_hash_function()
        return {"hash": hash_function(data_str.encode())}
    
    def _get_system_info(self):
        """Get system information."""
        return {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine()
            },
            "opacus_version": opacus.__version__,
            "timestamp": self.timestamp
        }
    
    def track_data(self, train_data, test_data):
        """Track data provenance."""
        self.logger.info("Tracking data provenance...")
        
        # Calculate basic statistics
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        train_min = np.min(train_data)
        train_max = np.max(train_data)
        
        test_mean = np.mean(test_data)
        test_std = np.std(test_data)
        test_min = np.min(test_data)
        test_max = np.max(test_data)
        
        # For train data
        train_stats = {
            "mean": train_mean,
            "std": train_std,
            "min": float(train_min),
            "max": float(train_max),
            "shape": list(train_data.shape),
            "dtype": str(train_data.dtype)
        }
        
        # For test data
        test_stats = {
            "mean": test_mean,
            "std": test_std,
            "min": float(test_min),
            "max": float(test_max),
            "shape": list(test_data.shape),
            "dtype": str(test_data.dtype)
        }
        
        # Generate data hashes
        train_hash = self._generate_hash(train_data)
        test_hash = self._generate_hash(test_data)
        
        # Create a comprehensive data info dictionary for hash computation
        data_info = {
            "dataset": "MNIST",
            "timestamp": self.timestamp,
            "train": {
                "samples": len(train_data),
                "mean": train_mean,
                "std": train_std,
                "hash": train_hash["hash"],
                "metadata": {
                    "statistics": train_stats
                }
            },
            "test": {
                "samples": len(test_data),
                "mean": test_mean,
                "std": test_std,
                "hash": test_hash["hash"],
                "metadata": {
                    "statistics": test_stats
                }
            }
        }
        
        # Log the data_info dictionary before hashing
        self.logger.info(f"[TRACKER] data_info to be hashed: {data_info}")
        data_hash = self._generate_hash(data_info, "data")
        self.logger.info(f"[TRACKER] data hash: {data_hash['hash']}")
        
        # Store data provenance with the structure expected by the verifier
        self.data["data_provenance"] = {
            "train": {
                "samples": len(train_data),
                "mean": train_mean,
                "std": train_std,
                "hash": train_hash["hash"],
                "metadata": {
                    "statistics": train_stats
                }
            },
            "test": {
                "samples": len(test_data),
                "mean": test_mean,
                "std": test_std,
                "hash": test_hash["hash"],
                "metadata": {
                    "statistics": test_stats
                }
            },
            "dataset": "MNIST",
            "timestamp": self.timestamp,
            "hashes": {
                "data": data_hash["hash"]
            }
        }
        
        # Store data hashes in the hashes section for backward compatibility
        self.data["hashes"]["train_data"] = train_hash["hash"]
        self.data["hashes"]["test_data"] = test_hash["hash"]
        self.data["hashes"]["data"] = data_hash["hash"]
        
        self.logger.info("Data provenance tracked successfully.")
    
    def track_model(self, model):
        """Track model architecture (which doesn't change during training)."""
        self.logger.info("Tracking model architecture...")
        
        # Get model architecture and parameters
        model_info = {
            "architecture": {
                "name": model.__class__.__name__,
                "layers": [
                    {
                        "name": name,
                        "type": layer.__class__.__name__,
                        "parameters": sum(p.numel() for p in layer.parameters())
                    }
                    for name, layer in model.named_children()
                ],
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "timestamp": self.timestamp
            }
        }
        
        # Store architecture info
        self.data["model_provenance"] = {
            "architecture": model_info["architecture"],
            "timestamp": self.timestamp
        }
        
        self.logger.info("Model architecture tracked successfully.")
    
    def track_training(self, model, config, final_metrics, training_logs=None, privacy_metrics=None):
        """Track training provenance."""
        self.logger.info("Tracking training process...")
        
        # Generate training hash
        training_hash = self._generate_hash({
            "config": config,
            "metrics": final_metrics,
            "privacy_metrics": privacy_metrics
        })
        
        # Generate privacy metrics hash if available
        privacy_hash = None
        if privacy_metrics:
            privacy_hash = self._generate_hash(privacy_metrics, "privacy_metrics")
        
        # Prepare privacy metrics with hash
        privacy_metrics_with_hash = privacy_metrics or {}
        if privacy_hash:
            privacy_metrics_with_hash["hash"] = privacy_hash["hash"]
        
        self.data["training_provenance"] = {
            "hash": training_hash["hash"],
            "config": config,
            "training_logs": training_logs or [],  # Store training logs if provided
            "final_metrics": final_metrics,
            "privacy_metrics": privacy_metrics_with_hash,
            "timestamp": self.timestamp
        }
        
        # Store hashes in the hashes section
        self.data["hashes"]["training"] = training_hash["hash"]
        if privacy_hash:
            self.data["hashes"]["privacy"] = privacy_hash["hash"]
        
        self.logger.info("Training provenance tracked successfully.")
    
    def update_training_provenance(self, epoch_data: Dict[str, Any]) -> None:
        """Update training provenance with epoch data."""
        if 'training_logs' not in self.data['training_provenance']:
            self.data['training_provenance']['training_logs'] = []
        
        # Add epoch data to training logs
        self.data['training_provenance']['training_logs'].append(epoch_data)
        
        # Create epoch node for Merkle tree
        epoch_node = {
            'epoch': epoch_data['epoch'],
            'metrics': {
                'train_loss': epoch_data['train_loss'],
                'val_loss': epoch_data['val_loss'],
                'train_acc': epoch_data['train_acc'],
                'val_acc': epoch_data['val_acc']
            },
            'model_state': epoch_data['model_state']
        }
        
        # Add privacy metrics if available
        if 'privacy_metrics' in epoch_data:
            epoch_node['privacy_metrics'] = epoch_data['privacy_metrics']
        
        # Add epoch node to Merkle tree
        self.merkle_tree.add_epoch_node(epoch_data['epoch'], epoch_node)
        
        # If this is the final epoch, update final metrics
        if epoch_data.get('is_final', False):
            self.data['training_provenance']['final_metrics'] = {
                'train_loss': epoch_data['train_loss'],
                'val_loss': epoch_data['val_loss'],
                'train_acc': epoch_data['train_acc'],
                'val_acc': epoch_data['val_acc']
            }
            if 'privacy_metrics' in epoch_data:
                self.data['training_provenance']['final_metrics']['privacy_metrics'] = epoch_data['privacy_metrics']
            
            # Note: track_training_run is called explicitly in the main training script
            # to avoid redundant Merkle tree dumps
    
    def save(self):
        """Save provenance data to JSON file."""
        self.logger.info("Saving provenance data...")
        
        # Get Merkle root if available
        merkle_root = None
        merkle_tree_file = None
        if hasattr(self, 'merkle_tree') and self.merkle_tree.root is not None:
            try:
                merkle_root = self.merkle_tree.get_root_hash()
                merkle_tree_file = getattr(self, 'merkle_tree_file', None)
            except Exception as e:
                self.logger.warning(f"Could not get Merkle root: {e}")
        
        # Prepare data for saving
        save_data = {
            "timestamp": self.timestamp,
            "system_info": self._make_serializable(self.data["system_info"]),
            "training_config": self._make_serializable(self.data.get("training_config", {})),
            "data_provenance": self._make_serializable(self.data["data_provenance"]),
            "model_provenance": self._make_serializable(self.data["model_provenance"]),
            "training_provenance": self._make_serializable(self.data.get("training_provenance", {})),
            "safety_metrics": self._make_serializable(self.data.get("safety_metrics", {})),
            "epoch_metrics": self._make_serializable(self.data.get("epoch_metrics", [])),
            "hashes": self._make_serializable(self.data["hashes"])
        }
        
        # Add Merkle tree information if available
        if merkle_root:
            save_data["merkle_tree"] = {
                "root_hash": merkle_root,
                "tree_file": merkle_tree_file,
                "algorithm": "blake3"  # or get from config
            }
        
        # Save to JSON file
        provenance_file = self.provenance_dir / "provenance_report.json"
        with open(provenance_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Provenance data saved to {provenance_file}")
        return str(provenance_file)
    
    def get_provenance_proof(self, component_type, component_data):
        """Generate a Merkle proof for a specific component."""
        if component_type not in ["data", "model", "training"]:
            raise ValueError(f"Invalid component: {component_type}")
        
        return {
            "component": component_type,
            "hash": self._generate_hash(component_data)["hash"],
            "timestamp": self.timestamp,
            "proof": self._generate_merkle_proof(component_type, component_data)
        }
    
    def _generate_merkle_proof(self, component_type, data):
        """Generate a Merkle proof for the given component and data."""
        # This is a simplified version. In a real implementation,
        # this would generate a proper Merkle proof with sibling hashes
        return [self._generate_hash(data)["hash"]]

    def _dump_merkle_tree(self):
        """Dump the Merkle tree structure to a JSON file."""
        tree_dict = self.merkle_tree.get_tree_dict()
        serializable_tree = convert_to_serializable(tree_dict)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"merkle_tree_{timestamp}.json"
        filepath = os.path.join(self.provenance_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(serializable_tree, f, indent=2)
        self.logger.info(f"Merkle tree dumped to {filepath}")
        
        # Store the filename for inclusion in the main report
        self.merkle_tree_file = filename

    def track_training_run(self, data_provenance, model_provenance, training_provenance):
        """Track a complete training run with detailed logging."""
        self.logger.info("Starting training run tracking...")
        
        # Build the hierarchical Merkle tree
        self.merkle_tree.build_provenance_tree(
            data_provenance,
            model_provenance,
            training_provenance
        )
        
        # Store the root hash
        root_hash = self.merkle_tree.get_root_hash()
        self.logger.info(f"Root hash: {root_hash}")
        
        # Get component hashes from the tree
        data_node = self.merkle_tree.get_node("data")
        model_node = self.merkle_tree.get_node("model")
        training_node = self.merkle_tree.get_node("training")
        
        # Store component hashes
        self.data["hashes"] = {
            "data": data_node["hash"] if data_node else None,
            "model": model_node["hash"] if model_node else None,
            "training": training_node["hash"] if training_node else None,
            "root": root_hash
        }
        
        # Store the full provenance data
        self.data["data_provenance"] = data_provenance
        self.data["model_provenance"] = model_provenance
        self.data["training_provenance"] = training_provenance
        
        # Generate proofs for each component
        proofs = {}
        for component in ["data", "model", "training"]:
            proof = self.merkle_tree.get_component_proof(component)
            if proof:
                proofs[component] = proof
        
        # Store proofs
        self.data["proofs"] = proofs
        
        self.logger.info("Training run tracking completed")
        
        # Dump Merkle tree for debugging
        self._dump_merkle_tree()
        
        return root_hash

    def set_final_metrics(self, final_metrics, config=None):
        """Set final metrics and save provenance data."""
        # Convert all nested values to serializable format
        serialized_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, dict):
                serialized_metrics[key] = convert_to_serializable(value)
            else:
                serialized_metrics[key] = convert_to_serializable(value)
        
        self.data["training_provenance"]["final_metrics"] = serialized_metrics
        if config:
            self.data["training_provenance"]["config"] = convert_to_serializable(config)
        self.save()
    
    def set_log_file_path(self, log_file_path: str):
        """Set the log file path in provenance data."""
        if "training_provenance" not in self.data:
            self.data["training_provenance"] = {}
        
        self.data["training_provenance"]["log_file_path"] = log_file_path
        self.logger.info(f"Log file path tracked: {log_file_path}")
    
    def store_merkle_on_blockchain_before_training(self, training_config: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Store Merkle root hash on blockchain before training begins.
        
        Args:
            training_config: Training configuration
            
        Returns:
            Dictionary mapping networks to transaction IDs, or None if blockchain not enabled
        """
        if not self.blockchain_tracker:
            self.logger.warning("Blockchain tracking not enabled - skipping pre-training storage")
            return None
        
        try:
            # Get current Merkle root hash
            if hasattr(self.merkle_tree, 'root') and self.merkle_tree.root is not None:
                merkle_root_hash = self.merkle_tree.get_root_hash()
            else:
                # Create initial Merkle tree with current data
                self.merkle_tree.build_provenance_tree(
                    self.data.get("data_provenance", {}),
                    self.data.get("model_provenance", {}),
                    self.data.get("training_provenance", {})
                )
                merkle_root_hash = self.merkle_tree.get_root_hash()
            
            # Store on blockchain
            transaction_ids = self.blockchain_tracker.store_before_training(
                merkle_root_hash, training_config
            )
            
            # Store blockchain info in provenance data
            self.data["blockchain"] = {
                "before_training": {
                    "merkle_root_hash": merkle_root_hash,
                    "transaction_ids": transaction_ids,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Stored pre-training Merkle root on blockchain: {merkle_root_hash}")
            return transaction_ids
            
        except Exception as e:
            self.logger.error(f"Failed to store pre-training hash on blockchain: {e}")
            return None
    
    def store_merkle_on_blockchain_after_training(self, training_results: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Store Merkle root hash on blockchain after training completes.
        
        Args:
            training_results: Training results and metrics
            
        Returns:
            Dictionary mapping networks to transaction IDs, or None if blockchain not enabled
        """
        if not self.blockchain_tracker:
            self.logger.warning("Blockchain tracking not enabled - skipping post-training storage")
            return None
        
        try:
            # Use existing Merkle tree if it's already built (from track_training_run)
            if hasattr(self.merkle_tree, 'root') and self.merkle_tree.root is not None:
                merkle_root_hash = self.merkle_tree.get_root_hash()
                self.logger.info(f"Using existing Merkle tree with root hash: {merkle_root_hash}")
            else:
                # Fallback: rebuild tree if it doesn't exist (shouldn't happen in normal flow)
                self.logger.warning("Merkle tree not found, rebuilding...")
                self.merkle_tree.build_provenance_tree(
                    self.data.get("data_provenance", {}),
                    self.data.get("model_provenance", {}),
                    self.data.get("training_provenance", {})
                )
                merkle_root_hash = self.merkle_tree.get_root_hash()
            
            # Store on blockchain
            transaction_ids = self.blockchain_tracker.store_after_training(
                merkle_root_hash, training_results
            )
            
            # Update blockchain info in provenance data
            if "blockchain" not in self.data:
                self.data["blockchain"] = {}
            
            self.data["blockchain"]["after_training"] = {
                "merkle_root_hash": merkle_root_hash,
                "transaction_ids": transaction_ids,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Stored post-training Merkle root on blockchain: {merkle_root_hash}")
            return transaction_ids
            
        except Exception as e:
            self.logger.error(f"Failed to store post-training hash on blockchain: {e}")
            return None
    
    def verify_blockchain_provenance(self) -> Dict[str, Any]:
        """
        Verify the complete provenance chain on blockchain.
        
        Returns:
            Dictionary containing verification results
        """
        if not self.blockchain_tracker:
            return {
                "blockchain_enabled": False,
                "verification_results": {},
                "chain_integrity": False
            }
        
        try:
            verification_results = self.blockchain_tracker.verify_provenance_chain()
            verification_results["blockchain_enabled"] = True
            
            self.logger.info("Blockchain provenance verification completed")
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Failed to verify blockchain provenance: {e}")
            return {
                "blockchain_enabled": True,
                "verification_results": {},
                "chain_integrity": False,
                "error": str(e)
            }
    
    def save_blockchain_report(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Save blockchain report to file.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Path to the saved report, or None if blockchain not enabled
        """
        if not self.blockchain_tracker:
            self.logger.warning("Blockchain tracking not enabled - cannot save blockchain report")
            return None
        
        try:
            if output_path is None:
                output_path = self.provenance_dir / "blockchain_report.json"
            
            report_path = self.blockchain_tracker.save_blockchain_report(output_path)
            self.logger.info(f"Blockchain report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to save blockchain report: {e}")
            return None
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """
        Get current blockchain status and configuration.
        
        Returns:
            Dictionary containing blockchain status
        """
        status = {
            "blockchain_enabled": self.blockchain_tracker is not None,
            "stored_hashes": {},
            "verification_results": {}
        }
        
        if self.blockchain_tracker:
            # Get stored hashes
            status["stored_hashes"] = self.blockchain_tracker.stored_hashes
            
            # Get verification results
            status["verification_results"] = self.verify_blockchain_provenance()
            
            # Get blockchain config
            status["blockchain_config"] = self.blockchain_tracker.config.get("blockchain", {})
        
        return status 