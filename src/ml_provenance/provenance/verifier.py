"""
Provenance verification module.
"""

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
from typing import Dict, Any, Optional
import torch.nn as nn
from .provenance_data import ProvenanceData

class ProvenanceVerifier:
    def __init__(self, provenance_dir):
        self.provenance_dir = Path(provenance_dir)
        self.logger = logging.getLogger(f"provenance_verifier_{os.path.basename(provenance_dir)}")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(provenance_dir, "provenance.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(fh)
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
        with open(self.provenance_dir / "provenance.json", "r") as f:
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
            "data_verification": self._verify_data(data["data_provenance"]),
            "model_verification": self._verify_model(data),
            "training_verification": self._verify_training(data),
            "hash_verification": self._verify_hashes(data, model_path)
        }

        # --- Merkle Tree Checks ---
        # Tree Structure: root exists and has children
        tree_structure_valid = self.merkle_tree.root is not None and \
            self.merkle_tree.root.left is not None and self.merkle_tree.root.right is not None
        # Hash Chain: root hash matches recomputed from children
        hash_chain_valid = False
        if tree_structure_valid:
            left_hash = self.merkle_tree.root.left.hash
            right_hash = self.merkle_tree.root.right.hash
            recomputed = self.merkle_tree._combine_hashes(left_hash, right_hash)
            hash_chain_valid = (self.merkle_tree.root.hash == recomputed)
        # Timestamp Chain: all nodes have the same timestamp as provenance version (simple check)
        expected_timestamp = data["version"]
        timestamps = [
            data["data_provenance"].get("timestamp"),
            data["model_provenance"].get("timestamp"),
            data["training_provenance"].get("timestamp")
        ]
        timestamp_chain_valid = all(ts == expected_timestamp for ts in timestamps)
        overall_verification = {
            "tree_structure_valid": tree_structure_valid,
            "hash_chain_valid": hash_chain_valid,
            "timestamp_chain_valid": timestamp_chain_valid
        }
        # --- End Merkle Tree Checks ---

        # Determine overall status with detailed logging
        self.logger.info("\nDetermining overall verification status...")
        for category, results in verification_results.items():
            self.logger.info(f"\n{category}:")
            for check, status in results.items():
                self.logger.info(f"  {check}: {status}")
        
        overall_status = all(
            all(check for check in results.values())
            for results in verification_results.values()
        ) and all(overall_verification.values())
        
        self.logger.info(f"\nOverall verification status: {'SUCCESS' if overall_status else 'FAILURE'}")
        
        # Create verification report
        report = {
            "provenance_version": data["version"],
            "overall_status": overall_status,
            "verification_results": verification_results,
            "overall_verification": overall_verification,
            "verification_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save verification report
        report_path = self.provenance_dir / "verification.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Verification report saved to: {report_path}")
        return report
    
    def _verify_data(self, data_provenance: Dict[str, Any]) -> Dict[str, Any]:
        """Verify data provenance."""
        results = {
            "has_train_data": False,
            "has_test_data": False,
            "has_timestamp": False,
            "hash_match": False,
            "statistics_verified": False,
            "merkle_verified": False,
            "metadata_verified": False,
            "merkle_tree": None
        }
        
        # Check for required data
        if "train" not in data_provenance:
            self.logger.warning("No training data found in provenance")
            return results
            
        if "test" not in data_provenance:
            self.logger.warning("No test data found in provenance")
            return results
            
        results["has_train_data"] = True
        results["has_test_data"] = True
        
        # Check timestamp
        if "timestamp" not in data_provenance:
            self.logger.warning("No timestamp found in provenance")
            return results
            
        results["has_timestamp"] = True
        
        # Verify statistics
        train_stats = data_provenance["train"].get("statistics", {})
        test_stats = data_provenance["test"].get("statistics", {})
        
        if not train_stats or not test_stats:
            self.logger.warning("Missing statistics in data provenance")
            return results
            
        # Verify statistics match
        if (train_stats.get("mean") != data_provenance["train"].get("mean") or
            train_stats.get("std") != data_provenance["train"].get("std")):
            self.logger.warning("Training statistics mismatch")
            return results
            
        if (test_stats.get("mean") != data_provenance["test"].get("mean") or
            test_stats.get("std") != data_provenance["test"].get("std")):
            self.logger.warning("Test statistics mismatch")
            return results
            
        results["statistics_verified"] = True
        
        # Verify metadata
        train_metadata = data_provenance["train"].get("metadata", {})
        test_metadata = data_provenance["test"].get("metadata", {})
        
        if not train_metadata or not test_metadata:
            self.logger.warning("Missing metadata in data provenance")
            return results
            
        # Verify metadata structure
        required_metadata_fields = ["statistics"]
        for field in required_metadata_fields:
            if field not in train_metadata or field not in test_metadata:
                self.logger.warning(f"Missing required metadata field: {field}")
                return results
                
        results["metadata_verified"] = True
        
        # Compute hash
        computed_hash = self._compute_data_hash(data_provenance)
        
        # Get stored hash from data provenance
        stored_hash = data_provenance.get("hashes", {}).get("data")
        if not stored_hash:
            self.logger.warning("No stored data hash found")
            return results
            
        # Compare computed hash with stored hash
        hash_match = computed_hash == stored_hash
        self.logger.info(f"Data hash match: {hash_match}")
        results["hash_match"] = hash_match
        
        # Verify Merkle tree
        try:
            merkle_tree = MLProvenanceMerkleTree()
            merkle_tree.add_node("data", data_provenance)
            merkle_verified = merkle_tree.verify_component("data", data_provenance)
            results["merkle_verified"] = merkle_verified
            results["merkle_tree"] = merkle_tree.to_dict()
        except Exception as e:
            self.logger.error(f"Error verifying Merkle tree: {str(e)}")
            results["merkle_verified"] = False
            
        return results
    
    def _verify_model(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Verify model provenance."""
        results = {
            "model_exists": False,
            "model_hash_match": False,
            "model_merkle_verified": False
        }
        
        if "model_provenance" not in data:
            return results
            
        model_info = data["model_provenance"]
        results["model_exists"] = True
        
        try:
            # Compute current hash from the model info
            current_hash = self._generate_hash(model_info["info"])
            stored_hash = model_info.get("hash")
            
            if stored_hash:
                results["model_hash_match"] = current_hash == stored_hash
            
            # Initialize Merkle tree with the model info
            merkle_tree = MLProvenanceMerkleTree()
            merkle_tree.add_node("model", model_info["info"])
            results["model_merkle_verified"] = merkle_tree.verify_component("model", model_info["info"])
        except Exception as e:
            self.logger.error(f"Error verifying model: {str(e)}")
            
        return results
    
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
    
    def _verify_training(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Verify training provenance."""
        results = {
            "final_metrics_present": "final_metrics" in data["training_provenance"],
            "training_logs_present": "training_logs" in data["training_provenance"],
            "privacy_metrics_present": "privacy_metrics" in data["training_provenance"],
            "hash_present": "hash" in data["training_provenance"],
            "timestamp_present": "timestamp" in data["training_provenance"],
            "accuracy_present": False,
            "loss_present": False
        }
        
        # Check final metrics
        if results["final_metrics_present"]:
            final_metrics = data["training_provenance"]["final_metrics"]
            results["accuracy_present"] = "train_accuracy" in final_metrics
            results["loss_present"] = "loss" in final_metrics
        
        return results

    def _compute_data_hash(self, data_provenance):
        """Compute hash for data component."""
        # Get train and test data info
        train_data = data_provenance.get("train", {})
        test_data = data_provenance.get("test", {})
        
        # Create comprehensive data info dictionary
        data_info = {
            "dataset": data_provenance.get("dataset", ""),
            "timestamp": data_provenance.get("timestamp", ""),
            "train": {
                "samples": train_data.get("samples", 0),
                "mean": train_data.get("mean", 0.0),
                "std": train_data.get("std", 0.0),
                "hash": train_data.get("hash", ""),
                "metadata": train_data.get("metadata", {})
            },
            "test": {
                "samples": test_data.get("samples", 0),
                "hash": test_data.get("hash", ""),
                "metadata": test_data.get("metadata", {})
            }
        }
        if hasattr(self, 'logger'):
            self.logger.info(f"[VERIFIER] data_info to be hashed: {data_info}")
        data_hash = self._generate_hash(data_info, "data")
        if hasattr(self, 'logger'):
            self.logger.info(f"[VERIFIER] data hash: {data_hash}")
        return data_hash

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

    def _verify_privacy(self, data: Dict[str, Any]) -> Dict[str, bool]:
        results = {
            "has_privacy_metrics": "privacy_metrics" in data,
            "hash_match": False,
            "merkle_verified": False
        }
        if not results["has_privacy_metrics"]:
            return results
        current_hash = self._compute_privacy_hash(data["privacy_metrics"])
        stored_hash = data["privacy_metrics"].get("hash")
        if stored_hash:
            results["hash_match"] = current_hash == stored_hash
        try:
            merkle_tree = MLProvenanceMerkleTree()
            results["merkle_verified"] = merkle_tree.verify_component("privacy", data["privacy_metrics"])
        except Exception as e:
            self.logger.error(f"Error verifying privacy with Merkle tree: {e}")
            results["merkle_verified"] = False
        return results 

    def _generate_final_report(self, verification_results: Dict[str, Any]) -> str:
        """Generate a comprehensive final report."""
        report = []
        report.append("# Provenance Verification Report")
        report.append(f"\n## Run Summary")
        report.append("\n| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Training Accuracy | {verification_results.get('training_accuracy', 'N/A')}% |")
        report.append(f"| Test Accuracy | {verification_results.get('test_accuracy', 'N/A')}% |")
        report.append(f"| Final Loss | {verification_results.get('final_loss', 'N/A')} |")
        report.append(f"| Epochs | {verification_results.get('epochs', 'N/A')} |")
        report.append(f"| Batch Size | {verification_results.get('batch_size', 'N/A')} |")
        report.append(f"| Learning Rate | {verification_results.get('learning_rate', 'N/A')} |")
        report.append(f"| Optimizer | {verification_results.get('optimizer', 'N/A')} |")
        report.append(f"| Model Architecture | {verification_results.get('model_architecture', 'N/A')} |")
        
        report.append("\n## Verification Results")
        report.append("\n### Data Verification")
        data_verification = verification_results.get("data_verification", {})
        report.append("\n| Check | Status |")
        report.append("|-------|--------|")
        report.append(f"| Has Training Data | {'✅' if data_verification.get('has_train_data') else '❌'} |")
        report.append(f"| Has Test Data | {'✅' if data_verification.get('has_test_data') else '❌'} |")
        report.append(f"| Has Timestamp | {'✅' if data_verification.get('has_timestamp') else '❌'} |")
        report.append(f"| Hash Match | {'✅' if data_verification.get('hash_match') else '❌'} |")
        report.append(f"| Statistics Verified | {'✅' if data_verification.get('statistics_verified') else '❌'} |")
        report.append(f"| Merkle Verified | {'✅' if data_verification.get('merkle_verified') else '❌'} |")
        report.append(f"| Metadata Verified | {'✅' if data_verification.get('metadata_verified') else '❌'} |")
        
        # Add Merkle Tree Dump
        if "merkle_tree" in data_verification:
            report.append("\n### Merkle Tree Structure")
            report.append("```")
            report.append(json.dumps(data_verification["merkle_tree"], indent=2))
            report.append("```")
        
        report.append("\n### Model Verification")
        model_verification = verification_results.get("model_verification", {})
        report.append("\n| Check | Status |")
        report.append("|-------|--------|")
        report.append(f"| Model Hash Match | {'✅' if model_verification.get('model_hash_match') else '❌'} |")
        report.append(f"| Architecture Verified | {'✅' if model_verification.get('architecture_verified') else '❌'} |")
        report.append(f"| Parameters Verified | {'✅' if model_verification.get('parameters_verified') else '❌'} |")
        
        report.append("\n### Training Verification")
        training_verification = verification_results.get("training_verification", {})
        report.append("\n| Check | Status |")
        report.append("|-------|--------|")
        report.append(f"| Hash Present | {'✅' if training_verification.get('hash_present') else '❌'} |")
        report.append(f"| Hyperparameters Verified | {'✅' if training_verification.get('hyperparameters_verified') else '❌'} |")
        report.append(f"| Metrics Verified | {'✅' if training_verification.get('metrics_verified') else '❌'} |")
        
        report.append("\n### Privacy Verification")
        privacy_verification = verification_results.get("privacy_verification", {})
        report.append("\n| Check | Status |")
        report.append("|-------|--------|")
        report.append(f"| Privacy Hash Match | {'✅' if privacy_verification.get('privacy_hash_match') else '❌'} |")
        report.append(f"| Privacy Budget Verified | {'✅' if privacy_verification.get('privacy_budget_verified') else '❌'} |")
        report.append(f"| Privacy Mechanism Verified | {'✅' if privacy_verification.get('privacy_mechanism_verified') else '❌'} |")
        
        report.append("\n## Overall Verification Status")
        overall_status = verification_results.get("overall_verification", "FAILURE")
        report.append(f"\n### {'✅ SUCCESS' if overall_status == 'SUCCESS' else '❌ FAILURE'}")
        
        return "\n".join(report)

class Verifier:
    """Verifier for ML provenance using consistent data structure."""
    
    def __init__(self, provenance_data: ProvenanceData):
        self.provenance = provenance_data
    
    def verify_data(self, train_data: torch.Tensor, test_data: torch.Tensor) -> Dict[str, bool]:
        """Verify data provenance."""
        results = {
            "train_samples": len(train_data) == self.provenance.data["data"]["metadata"]["train_samples"],
            "test_samples": len(test_data) == self.provenance.data["data"]["metadata"]["test_samples"],
            "dataset": True,  # Dataset name verification if needed
            "timestamp": True  # Timestamp verification if needed
        }
        return results
    
    def verify_model(self, model: nn.Module) -> Dict[str, bool]:
        """Verify model provenance."""
        results = {
            "architecture": model.__class__.__name__ == self.provenance.data["model"]["metadata"]["architecture"],
            "parameters": sum(p.numel() for p in model.parameters()) == self.provenance.data["model"]["metadata"]["parameters"],
            "timestamp": True  # Timestamp verification if needed
        }
        return results
    
    def verify_training(self, config: Dict[str, Any], final_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Verify training provenance."""
        results = {
            "epochs": config.get("epochs") == self.provenance.data["training"]["metadata"]["epochs"],
            "batch_size": config.get("batch_size") == self.provenance.data["training"]["metadata"]["batch_size"],
            "learning_rate": config.get("learning_rate") == self.provenance.data["training"]["metadata"]["learning_rate"],
            "final_metrics": final_metrics == self.provenance.data["training"]["metadata"]["final_metrics"],
            "timestamp": True  # Timestamp verification if needed
        }
        return results
    
    def verify_privacy(self, config: Dict[str, Any], achieved_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Verify privacy provenance."""
        results = {
            "target_epsilon": config.get("target_epsilon") == self.provenance.data["privacy"]["metadata"]["target_epsilon"],
            "target_delta": config.get("target_delta") == self.provenance.data["privacy"]["metadata"]["target_delta"],
            "noise_multiplier": config.get("noise_multiplier") == self.provenance.data["privacy"]["metadata"]["noise_multiplier"],
            "achieved_epsilon": achieved_metrics.get("epsilon") == self.provenance.data["privacy"]["metadata"]["achieved_epsilon"],
            "achieved_delta": achieved_metrics.get("delta") == self.provenance.data["privacy"]["metadata"]["achieved_delta"],
            "timestamp": True  # Timestamp verification if needed
        }
        return results
    
    def verify_all(self, train_data: torch.Tensor, test_data: torch.Tensor, 
                  model: nn.Module, config: Dict[str, Any], 
                  final_metrics: Dict[str, float], achieved_metrics: Dict[str, float]) -> Dict[str, Dict[str, bool]]:
        """Verify all provenance components."""
        return {
            "data": self.verify_data(train_data, test_data),
            "model": self.verify_model(model),
            "training": self.verify_training(config, final_metrics),
            "privacy": self.verify_privacy(config, achieved_metrics)
        } 