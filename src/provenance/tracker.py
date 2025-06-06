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
    
    def _generate_hash(self, data, component_name=None):
        """Generate SHA-256 hash of the given data with detailed logging."""
        self.logger.info(f"Generating hash for {component_name or 'data'}")
        
        if isinstance(data, dict):
            # Convert tensors and ndarrays to lists in dictionaries
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    serializable_data[key] = value.cpu().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value
            data = serializable_data
        elif isinstance(data, torch.Tensor):
            data = data.cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            data = data.tolist()
        
        # Log the data structure being hashed
        self.logger.debug(f"Data structure for {component_name}: {json.dumps(data, indent=2)}")
        
        # Serialize to JSON with consistent formatting
        data = json.dumps(data, sort_keys=True, indent=2).encode('utf-8')
        hash_value = hashlib.sha256(data).hexdigest()
        
        self.logger.info(f"Generated hash for {component_name}: {hash_value}")
        return hash_value
    
    def _get_system_info(self):
        """Get system information."""
        return {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine()
            }
        }
    
    def track_data(self, train_data, test_data):
        """Track data provenance."""
        self.logger.info("Tracking data provenance...")
        
        # Calculate basic statistics
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        
        # Generate data hashes
        train_hash = self._generate_hash(train_data)
        test_hash = self._generate_hash(test_data)
        
        self.data["data_provenance"] = {
            "train": {
                "samples": len(train_data),
                "mean": float(train_mean),
                "std": float(train_std),
                "hash": train_hash
            },
            "test": {
                "samples": len(test_data),
                "hash": test_hash
            }
        }
        
        # Store data hashes in the hashes section
        self.data["hashes"]["train_data"] = train_hash
        self.data["hashes"]["test_data"] = test_hash
        
        self.logger.info("Data provenance tracked successfully.")
    
    def track_model(self, model):
        """Track model provenance."""
        self.logger.info("Tracking model provenance...")
        
        # Get model architecture and parameters
        model_config = {
            "layers": [
                {"name": name, "type": module.__class__.__name__}
                for name, module in model.named_modules()
                if len(list(module.children())) == 0
            ]
        }
        trainable_params = sum(p.numel() for p in model.parameters())
        
        # Generate model hash
        model_hash = self._generate_hash(model_config)
        
        # Generate weights hash
        weights_hash = self._generate_hash(model.state_dict())
        
        self.data["model_provenance"] = {
            "architecture": model_config,
            "parameters": {
                "trainable_params": trainable_params
            },
            "hashes": {
                "architecture": model_hash,
                "weights": weights_hash
            }
        }
        
        # Store model hashes in the hashes section
        self.data["hashes"]["model_architecture"] = model_hash
        self.data["hashes"]["model_weights"] = weights_hash
        
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
        
        self.data["training_provenance"] = {
            "config": config,
            "final_metrics": final_metrics,
            "hash": training_hash,
            "training_logs": training_logs or [],  # Store training logs if provided
            "privacy_metrics": privacy_metrics or {}  # Store privacy metrics if provided
        }
        
        # Store training hash in the hashes section
        self.data["hashes"]["training"] = training_hash
        
        self.logger.info("Training provenance tracked successfully.")
    
    def save(self):
        """Save provenance data to JSON file."""
        self.logger.info("Saving provenance data...")
        
        # Generate overall provenance hash using Merkle tree
        root_hash = self.merkle_tree.track_training_run(
            self.data["data_provenance"],
            self.data["model_provenance"],
            self.data["training_provenance"]
        )
        self.data["hashes"]["overall"] = root_hash
        
        # Save main provenance data
        with open(self.provenance_dir / "data.json", "w") as f:
            json.dump(self.data, f, indent=2)
        
        # Save business report
        business_report = {
            "timestamp": self.timestamp,
            "model_parameters": self.data["model_provenance"]["parameters"]["trainable_params"],
            "training_metrics": self.data["training_provenance"]["final_metrics"],
            "data_stats": {
                "train_samples": self.data["data_provenance"]["train"]["samples"],
                "test_samples": self.data["data_provenance"]["test"]["samples"]
            },
            "provenance_hash": root_hash
        }
        
        with open(self.provenance_dir / "business_report.json", "w") as f:
            json.dump(business_report, f, indent=2)
        
        self.logger.info("Provenance data saved successfully.")
    
    def get_provenance_proof(self, component_type, component_data):
        """Generate a Merkle proof for a specific component."""
        return self.merkle_tree.generate_proof({
            'type': component_type,
            'content': component_data
        })

    def track_training_run(self, data_provenance, model_provenance, training_provenance):
        """Track a complete training run with detailed logging."""
        self.logger.info("Starting training run tracking...")
        
        # Generate hashes for each component
        data_hash = self._generate_hash(data_provenance, "data")
        model_hash = self._generate_hash(model_provenance, "model")
        training_hash = self._generate_hash(training_provenance, "training")
        
        # Log hash generation
        self.logger.info("\nGenerated hashes:")
        self.logger.info(f"Data:     {data_hash}")
        self.logger.info(f"Model:    {model_hash}")
        self.logger.info(f"Training: {training_hash}")
        
        # Add to Merkle tree
        self.merkle_tree.add_node(data_hash)
        self.merkle_tree.add_node(model_hash)
        self.merkle_tree.add_node(training_hash)
        
        # Generate overall hash
        overall_hash = self._generate_hash({
            "data": data_hash,
            "model": model_hash,
            "training": training_hash
        }, "overall")
        
        self.logger.info(f"Overall hash: {overall_hash}")
        
        # Store hashes in provenance data
        self.data["hashes"] = {
            "data": data_hash,
            "model": model_hash,
            "training": training_hash,
            "overall": overall_hash
        }
        
        self.logger.info("Training run tracking completed")
        return overall_hash 