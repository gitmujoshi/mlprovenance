import json
import os
import sys
import platform
import git
from datetime import datetime
from pathlib import Path
import logging
import tensorflow as tf
import numpy as np
import hashlib
import pickle
import base64

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
    
    def _generate_hash(self, data):
        """Generate SHA-256 hash of the given data."""
        if isinstance(data, (dict, list)):
            # Convert numpy arrays and bytes in lists to base64 strings before JSON serialization
            if isinstance(data, list):
                data = [base64.b64encode(x.tobytes()).decode('utf-8') if isinstance(x, np.ndarray) else x for x in data]
            data = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, (np.ndarray, tf.Tensor)):
            if isinstance(data, tf.Tensor):
                data = data.numpy().tobytes()
            else:
                data = data.tobytes()
        elif not isinstance(data, bytes):
            data = str(data).encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def _get_system_info(self):
        """Get system information."""
        return {
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
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
        model_config = model.get_config()
        trainable_params = model.count_params()
        
        # Generate model hash
        model_hash = self._generate_hash(model_config)
        
        # Sort weights by layer name and weight type to ensure consistent ordering
        weights = []
        for layer in model.layers:
            for i, w in enumerate(layer.weights):
                # Convert to float32 to ensure consistent precision
                w_numpy = w.numpy().astype(np.float32)
                # Create a tuple of (layer_name, weight_type, weight_array)
                weight_type = 'kernel' if i == 0 else 'bias'
                weights.append((layer.name, weight_type, w_numpy))
        
        # Sort by layer name and weight type
        weights.sort(key=lambda x: (x[0], x[1]))
        
        # Log some debug information
        self.logger.debug(f"Number of weight tensors: {len(weights)}")
        for layer_name, weight_type, w in weights:
            self.logger.debug(f"Layer: {layer_name}, Type: {weight_type}, Shape: {w.shape}, Mean: {np.mean(w):.6f}")
        
        # Generate hash using only the numpy arrays
        weights_hash = self._generate_hash([w[2] for w in weights])
        
        self.data["model_provenance"] = {
            "architecture": model_config,
            "parameters": {
                "trainable_params": trainable_params,
                "optimizer": {
                    "name": model.optimizer.__class__.__name__,
                    "config": model.optimizer.get_config()
                }
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
    
    def track_training(self, config, final_metrics, training_logs=None):
        """Track training provenance."""
        self.logger.info("Tracking training provenance...")
        
        # Generate training hash
        training_hash = self._generate_hash({
            "config": config,
            "metrics": final_metrics
        })
        
        self.data["training_provenance"] = {
            "config": config,
            "final_metrics": final_metrics,
            "hash": training_hash,
            "training_logs": training_logs or []  # Store training logs if provided
        }
        
        # Store training hash in the hashes section
        self.data["hashes"]["training"] = training_hash
        
        self.logger.info("Training provenance tracked successfully.")
    
    def save(self):
        """Save provenance data to JSON file."""
        self.logger.info("Saving provenance data...")
        
        # Generate overall provenance hash
        overall_hash = self._generate_hash({
            "data": self.data["data_provenance"],
            "model": self.data["model_provenance"],
            "training": self.data["training_provenance"]
        })
        self.data["hashes"]["overall"] = overall_hash
        
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
            "provenance_hash": overall_hash
        }
        with open(self.provenance_dir / "business_report.json", "w") as f:
            json.dump(business_report, f, indent=2)
        
        self.logger.info("Provenance data saved successfully.") 