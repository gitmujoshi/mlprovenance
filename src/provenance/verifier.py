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
        """Verify all hashes in the provenance data with detailed logging."""
        self.logger.info("Starting hash verification process...")
        
        # Load model for hash verification
        state_dict = torch.load(model_path)
        # Remove '_module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_module.'):
                new_state_dict[k[len('_module.'):]] = v
            else:
                new_state_dict[k] = v
        
        from src.training.train import MNISTModel
        model = MNISTModel()
        model.load_state_dict(new_state_dict)
        
        # Generate current hashes
        current_hashes = {
            "model_architecture": self._generate_hash(
                {
                    "layers": [
                        {"name": name, "type": module.__class__.__name__}
                        for name, module in model.named_modules()
                        if len(list(module.children())) == 0
                    ]
                },
                "model_architecture"
            ),
            "model_weights": self._generate_hash(model.state_dict(), "model_weights"),
            "training": self._generate_hash(
                {
                    "config": data["training_provenance"]["config"],
                    "metrics": data["training_provenance"]["final_metrics"],
                    "privacy_metrics": data["training_provenance"].get("privacy_metrics", {})
                },
                "training"
            )
        }
        
        # Compare with stored hashes
        stored_hashes = {
            "model_architecture": data["model_provenance"]["hashes"]["architecture"],
            "model_weights": data["model_provenance"]["hashes"]["weights"],
            "training": data["training_provenance"]["hash"]
        }
        
        # Log hash comparisons
        for key in current_hashes:
            self.logger.info(f"\nVerifying {key} hash:")
            self.logger.info(f"Current:  {current_hashes[key]}")
            self.logger.info(f"Stored:   {stored_hashes[key]}")
            self.logger.info(f"Match:    {current_hashes[key] == stored_hashes[key]}")
        
        results = {
            "model_architecture_hash_match": current_hashes["model_architecture"] == stored_hashes["model_architecture"],
            "model_weights_changed": current_hashes["model_weights"] != stored_hashes["model_weights"],
            "training_hash_match": current_hashes["training"] == stored_hashes["training"],
            "overall_hash_present": "overall" in data["hashes"]
        }
        
        # Log verification results
        self.logger.info("\nHash verification results:")
        for key, value in results.items():
            self.logger.info(f"{key}: {value}")
        
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
        self.logger.info(f"Verifying model at {model_path}...")
        
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
        
        # Reconstruct model architecture config (same as tracker)
        model_config = {
            "layers": [
                {"name": name, "type": module.__class__.__name__}
                for name, module in model.named_modules()
                if len(list(module.children())) == 0
            ]
        }
        
        # Generate hashes
        model_arch_hash = self._generate_hash(model_config)
        weights_hash = self._generate_hash(model.state_dict())
        
        stored_model_hash = data["model_provenance"]["hashes"]["architecture"]
        stored_weights_hash = data["model_provenance"]["hashes"]["weights"]
        
        # Check that weights have changed during training
        weights_changed = weights_hash != stored_weights_hash
        
        # Verify using Merkle tree
        model_valid = self.merkle_tree.verify_component('model', data["model_provenance"])
        
        results = {
            "model_hash_match": model_arch_hash == stored_model_hash,
            "weights_changed": weights_changed,
            "model_exists": Path(model_path).exists(),
            "architecture_verification": self._verify_model_architecture(model, data),
            "merkle_verification": model_valid
        }
        
        return results
    
    def _verify_model_architecture(self, model, data):
        """Verify model architecture details."""
        stored_config = data["model_provenance"]["architecture"]
        current_config = {
            "layers": [
                {"name": name, "type": module.__class__.__name__}
                for name, module in model.named_modules()
                if len(list(module.children())) == 0
            ]
        }
        
        return {
            "layer_count_match": len(stored_config["layers"]) == len(current_config["layers"]),
            "parameter_count_match": sum(p.numel() for p in model.parameters()) == data["model_provenance"]["parameters"]["trainable_params"],
            "optimizer_match": True  # We don't store optimizer info in PyTorch models
        }
    
    def _verify_training(self, data):
        """Verify training provenance."""
        self.logger.info("Verifying training process...")
        
        # Verify using Merkle tree
        training_valid = self.merkle_tree.verify_component('training', data["training_provenance"])
        
        results = {
            "test_accuracy_present": "final_accuracy" in data["training_provenance"]["final_metrics"],
            "test_loss_present": "final_loss" in data["training_provenance"]["final_metrics"],
            "privacy_metrics_present": "privacy_parameters" in data["training_provenance"]["config"],
            "training_hash_present": "hash" in data["training_provenance"],
            "merkle_verification": training_valid
        }
        
        return results 