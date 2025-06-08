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
        """Track model provenance."""
        model_info = {
            "architecture": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "timestamp": datetime.now().isoformat()
        }
        self.provenance.update_model(model_info)
    
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
    def __init__(self, base_dir="artifacts"):
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
    
    def _make_serializable(self, data):
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
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
        return {"hash": hashlib.sha256(data_str.encode()).hexdigest()}
    
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
            "dataset": "MNIST",
            "timestamp": self.timestamp
        }
        
        # Store data hashes in the hashes section
        self.data["hashes"]["train_data"] = train_hash["hash"]
        self.data["hashes"]["test_data"] = test_hash["hash"]
        self.data["hashes"]["data"] = data_hash["hash"]
        
        self.logger.info("Data provenance tracked successfully.")
    
    def track_model(self, model):
        """Track model provenance."""
        self.logger.info("Tracking model provenance...")
        
        # Get model architecture and parameters
        model_info = {
            "architecture": model.__class__.__name__,
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
        
        # Generate model hash
        model_hash = self._generate_hash(model_info)
        
        self.data["model_provenance"] = {
            "hash": model_hash["hash"],
            "info": model_info,
            "timestamp": self.timestamp
        }
        
        # Store model hash in the hashes section
        self.data["hashes"]["model_architecture"] = model_hash["hash"]
        
        self.logger.info("Model provenance tracked successfully.")
    
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
        
        # Generate hash for model state
        model_state_hash = self._generate_hash(epoch_data['model_state'], 'model_state')
        
        # Create epoch node for Merkle tree
        epoch_node = {
            'epoch': epoch_data['epoch'],
            'metrics': {
                'loss': epoch_data['loss'],
                'train_accuracy': epoch_data['train_accuracy'],
                'test_accuracy': epoch_data['test_accuracy']
            },
            'model_state_hash': model_state_hash['hash']
        }
        
        # Add privacy metrics if available
        if 'privacy_metrics' in epoch_data:
            epoch_node['privacy_metrics'] = epoch_data['privacy_metrics']
            # Update the privacy metrics in training provenance
            self.data['training_provenance']['privacy_metrics'] = epoch_data['privacy_metrics']
        
        # Add epoch node to Merkle tree
        self.merkle_tree.add_node(f"epoch_{epoch_data['epoch']}", epoch_node)
        
        # If this is the final epoch, update final metrics
        if epoch_data.get('is_final', False):
            self.data['training_provenance']['final_metrics'] = {
                'loss': epoch_data['loss'],
                'train_accuracy': epoch_data['train_accuracy'],
                'test_accuracy': epoch_data['test_accuracy'],
                'model_state_hash': model_state_hash['hash']
            }
            if 'privacy_metrics' in epoch_data:
                self.data['training_provenance']['final_metrics']['privacy_metrics'] = epoch_data['privacy_metrics']
            
            # Build final Merkle tree
            self.track_training_run(
                self.data['data_provenance'],
                self.data['model_provenance'],
                self.data['training_provenance']
            )
    
    def save(self):
        """Save provenance data to JSON file."""
        self.logger.info("Saving provenance data...")
        
        # Prepare data for saving
        save_data = {
            "version": self.data["version"],
            "data_provenance": self._make_serializable(self.data["data_provenance"]),
            "model_provenance": self._make_serializable(self.data["model_provenance"]),
            "training_provenance": self._make_serializable({
                "epochs": self.data["training_provenance"].get("epochs", 0),
                "batch_size": self.data["training_provenance"].get("batch_size", 64),
                "learning_rate": self.data["training_provenance"].get("learning_rate", 0.001),
                "final_accuracy": self.data["training_provenance"].get("final_accuracy", 0.0),
                "final_loss": self.data["training_provenance"].get("final_loss", 0.0),
                "training_logs": self.data["training_provenance"].get("training_logs", []),
                "privacy_metrics": self.data["training_provenance"].get("privacy_metrics", {}),
                "final_metrics": convert_to_serializable(self.data["training_provenance"].get("final_metrics", {})),
                "timestamp": self.data["training_provenance"].get("timestamp", self.timestamp)
            }),
            "system_info": self._make_serializable(self.data["system_info"]),
            "hashes": self._make_serializable(self.data["hashes"])
        }
        
        # Save to JSON file
        provenance_file = self.provenance_dir / "provenance.json"
        with open(provenance_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Provenance data saved to {provenance_file}")
    
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
        filepath = os.path.join(self.provenance_dir, "merkle_tree_dump.json")
        with open(filepath, "w") as f:
            json.dump(serializable_tree, f, indent=2)
        self.logger.info(f"Merkle tree dumped to {filepath}")

    def track_training_run(self, data_provenance, model_provenance, training_provenance):
        """Track a complete training run with detailed logging."""
        self.logger.info("Starting training run tracking...")
        # Generate hashes for each component
        data_hash = self._generate_hash(data_provenance, "data")["hash"]
        model_hash = self._generate_hash(model_provenance, "model")["hash"]
        training_hash = self._generate_hash(training_provenance, "training")["hash"]
        # Log hash generation
        self.logger.info("\nGenerated hashes:")
        self.logger.info(f"Data:     {data_hash}")
        self.logger.info(f"Model:    {model_hash}")
        self.logger.info(f"Training: {training_hash}")
        # Ensure data is serializable before adding to Merkle tree
        serializable_data = self._make_serializable(data_provenance)
        serializable_model = self._make_serializable(model_provenance)
        serializable_training = self._make_serializable(training_provenance)
        # Add to Merkle tree with both node_id and serializable data
        self.merkle_tree.add_node("data", serializable_data)
        self.merkle_tree.add_node("model", serializable_model)
        self.merkle_tree.add_node("training", serializable_training)
        # Generate overall hash
        overall_hash = self._generate_hash({
            "data": data_hash,
            "model": model_hash,
            "training": training_hash
        }, "overall")["hash"]
        self.logger.info(f"Overall hash: {overall_hash}")
        # Store hashes in provenance data
        self.data["hashes"] = {
            "data": data_hash,
            "model": model_hash,
            "training": training_hash,
            "overall": overall_hash
        }
        self.logger.info("Training run tracking completed")
        # Dump Merkle tree for debugging
        self._dump_merkle_tree()
        return overall_hash

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