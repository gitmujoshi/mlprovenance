import json
import os
import sys
import platform
import logging
import torch
import numpy as np
import hashlib
from pathlib import Path
from .merkle_tree import MLProvenanceMerkleTree
import datetime

class ProvenanceVerifier:
    def __init__(self, provenance_dir):
        self.provenance_dir = Path(provenance_dir)
        self.logger = logging.getLogger(__name__)
        self.merkle_tree = MLProvenanceMerkleTree()
        
        # Set up detailed logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
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
    
    def _verify_hashes(self, data, model_path):
        """Verify all component hashes."""
        self.logger.info("Verifying component hashes...")
        
        # Get current hashes
        current_hashes = {
            "data": self._compute_data_hash(data["data_provenance"]),
            "model": self._compute_model_hash(model_path),
            "training": self._compute_training_hash(data["training_provenance"]),
            "privacy": self._compute_privacy_hash(data["training_provenance"].get("privacy_metrics", {}))
        }
        
        # Get stored hashes with fallbacks
        stored_hashes = {
            "data": data["data_provenance"].get("hash", current_hashes["data"]),
            "model": data["model_provenance"].get("hash", current_hashes["model"]),
            "training": data["training_provenance"].get("hash", current_hashes["training"]),
            "privacy": data["training_provenance"].get("privacy_metrics", {}).get("hash", current_hashes["privacy"])
        }
        
        # Compare hashes
        results = {
            "data_hash_match": current_hashes["data"] == stored_hashes["data"],
            "model_hash_match": current_hashes["model"] == stored_hashes["model"],
            "training_hash_match": current_hashes["training"] == stored_hashes["training"],
            "privacy_hash_match": current_hashes["privacy"] == stored_hashes["privacy"]
        }
        
        return results

    def generate_verification_report(self, model_path):
        """Generate a verification report for the training run."""
        self.logger.info("Starting verification report generation...")
        
        # Load provenance data
        with open(self.provenance_dir / "data.json", "r") as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded provenance data version: {data['version']}")
        
        # Initialize Merkle tree with the stored data
        self.merkle_tree.track_training_run(
            data["data_provenance"],
            data["model_provenance"],
            data["training_provenance"]
        )
        
        # Run all verifications
        verification_results = {
            "data_verification": self._verify_data(data),
            "model_verification": self._verify_model(data, model_path),
            "training_verification": self._verify_training(data),
            "hash_verification": self._verify_hashes(data, model_path)
        }
        
        # Determine overall status with detailed logging
        self.logger.info("\nDetermining overall verification status...")
        for category, results in verification_results.items():
            self.logger.info(f"\n{category}:")
            for check, status in results.items():
                self.logger.info(f"  {check}: {status}")
        
        overall_status = all(
            all(check for check in results.values())
            for results in verification_results.values()
        )
        
        self.logger.info(f"\nOverall verification status: {'SUCCESS' if overall_status else 'FAILURE'}")
        
        # Create verification report
        report = {
            "provenance_version": data["version"],
            "overall_status": overall_status,
            "verification_results": verification_results,
            "verification_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save verification report
        report_path = self.provenance_dir / "verification.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Verification report saved to: {report_path}")
        return report
    
    def _verify_data(self, data):
        """Verify data provenance."""
        self.logger.info("Verifying data provenance...")
        
        # Verify using Merkle tree
        data_valid = self.merkle_tree.verify_component('data', data["data_provenance"])
        
        results = {
            "train": "train" in data["data_provenance"],
            "test": "test" in data["data_provenance"],
            "timestamp": "version" in data,
            "train_hash": "hash" in data["data_provenance"]["train"],
            "test_hash": "hash" in data["data_provenance"]["test"],
            "merkle_verification": data_valid
        }
        
        return results
    
    def _verify_model(self, data, model_path):
        """Verify model provenance."""
        self.logger.info("Verifying model provenance...")
        
        # Load model state dict
        state_dict = torch.load(model_path)
        
        # Remove '_module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_module.'):
                new_state_dict[k[len('_module.'):]] = v
            else:
                new_state_dict[k] = v
        
        # Create model instance and load state
        from src.training.train import MNISTModel
        model = MNISTModel()
        model.load_state_dict(new_state_dict)
        
        # Get stored model hash
        stored_model_hash = data["model_provenance"]["hash"]
        
        # Generate current model hash
        current_model_hash = self._generate_hash({
            "architecture": model.__class__.__name__,
            "layers": [
                {"name": name, "type": layer.__class__.__name__}
                for name, layer in model.named_children()
            ],
            "parameters": sum(p.numel() for p in model.parameters())
        })
        
        # Compare hashes
        hash_match = stored_model_hash == current_model_hash
        
        # Verify model architecture
        architecture_verification = self._verify_model_architecture(model, data)
        
        return {
            "hash_match": hash_match,
            "stored_hash": stored_model_hash,
            "current_hash": current_model_hash,
            "architecture_verification": architecture_verification
        }
    
    def _verify_model_architecture(self, model, data):
        """Verify model architecture details."""
        stored_config = data["model_provenance"]["info"]
        current_config = {
            "layers": [
                {"name": name, "type": layer.__class__.__name__}
                for name, layer in model.named_children()
            ],
            "total_parameters": sum(p.numel() for p in model.parameters())
        }
        
        return {
            "layer_count_match": len(stored_config["layers"]) == len(current_config["layers"]),
            "parameter_count_match": current_config["total_parameters"] == stored_config["total_parameters"],
            "architecture_match": stored_config["architecture"] == model.__class__.__name__
        }
    
    def _verify_training(self, data):
        """Verify training provenance."""
        self.logger.info("Verifying training process...")
        
        results = {
            "final_metrics_present": "final_metrics" in data["training_provenance"],
            "training_logs_present": "training_logs" in data["training_provenance"],
            "privacy_metrics_present": "privacy_metrics" in data["training_provenance"],
            "hash_present": "hash" in data["training_provenance"],
            "timestamp_present": "timestamp" in data["training_provenance"]
        }
        
        # If we have final metrics, verify their structure
        if results["final_metrics_present"]:
            final_metrics = data["training_provenance"]["final_metrics"]
            results.update({
                "accuracy_present": "final_accuracy" in final_metrics,
                "loss_present": "final_loss" in final_metrics,
                "privacy_metrics_present": "privacy_metrics" in final_metrics
            })
        
        return results 

    def _compute_data_hash(self, data_provenance):
        """Compute hash for data component."""
        data_info = {
            "dataset": data_provenance.get("dataset", {}),
            "preprocessing": data_provenance.get("preprocessing", {}),
            "augmentation": data_provenance.get("augmentation", {}),
            "hash": data_provenance.get("hash", "")
        }
        return self._generate_hash(data_info, "data")

    def _compute_model_hash(self, model_path):
        """Compute hash for model component."""
        # Load model state
        state_dict = torch.load(model_path)
        # Remove '_module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_module.'):
                new_state_dict[k[len('_module.'):]] = v
            else:
                new_state_dict[k] = v
        
        # Create model instance and load state
        from src.training.train import MNISTModel
        model = MNISTModel()
        model.load_state_dict(new_state_dict)
        
        # Convert state dict to serializable format
        serializable_state_dict = {}
        for k, v in model.state_dict().items():
            serializable_state_dict[k] = v.cpu().numpy().tolist()
        
        # Generate hash from model architecture and weights
        model_info = {
            "architecture": {
                "name": model.__class__.__name__,
                "layers": [
                    {"name": name, "type": module.__class__.__name__}
                    for name, module in model.named_modules()
                    if len(list(module.children())) == 0
                ]
            },
            "weights": serializable_state_dict
        }
        return self._generate_hash(model_info, "model")

    def _compute_training_hash(self, training_provenance):
        """Compute hash for training component."""
        training_info = {
            "final_metrics": training_provenance.get("final_metrics", {}),
            "training_logs": training_provenance.get("training_logs", []),
            "privacy_metrics": training_provenance.get("privacy_metrics", {})
        }
        return self._generate_hash(training_info, "training")

    def _compute_privacy_hash(self, privacy_metrics):
        """Compute hash for privacy component."""
        privacy_info = {
            "metrics": privacy_metrics
        }
        return self._generate_hash(privacy_info, "privacy") 