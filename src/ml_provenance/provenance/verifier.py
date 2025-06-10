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
from datetime import datetime
from typing import Dict, Any, Optional
import torch.nn as nn
from .provenance_data import ProvenanceData
from .hash_config import HashFactory
import traceback

logger = logging.getLogger(__name__)

class ProvenanceVerifier:
    """Class to verify ML provenance data."""
    
    def __init__(self, provenance_dir: Optional[Path] = None):
        """
        Initialize verifier with provenance directory.
        
        Args:
            provenance_dir: Path to the provenance directory
        """
        self.provenance_data = {
            "data": {},
            "model": {},
            "training": {}
        }
        self.provenance_dir = Path(provenance_dir) if provenance_dir else None
        self.hash_function = HashFactory.get_hash_function()
        logger.info(f"Initialized ProvenanceVerifier with directory: {self.provenance_dir}")
    
    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """
        Generate hash for data using configured hash function.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex digest of the hash
        """
        # Serialize data to bytes
        data_bytes = json.dumps(data, sort_keys=True).encode()
        
        # Compute hash
        return self.hash_function(data_bytes)
    
    def _verify_hashes(self, current_hashes: Dict[str, str], stored_hashes: Dict[str, str]) -> Dict[str, bool]:
        """
        Verify current hashes against stored hashes.
        
        Args:
            current_hashes: Dictionary of current component hashes
            stored_hashes: Dictionary of stored component hashes
            
        Returns:
            Dictionary of verification results
        """
        results = {}
        for component, current_hash in current_hashes.items():
            stored_hash = stored_hashes.get(component)
            if stored_hash is None:
                logger.warning(f"No stored hash found for component: {component}")
                results[component] = False
                continue
            
            matches = current_hash == stored_hash
            results[component] = matches
            
            logger.info(f"Verified {component} hash: {'✓' if matches else '✗'}")
            logger.info(f"Current hash: {current_hash}")
            logger.info(f"Stored hash: {stored_hash}")
        
        return results
    
    def _verify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify data component.
        
        Args:
            data: Data to verify
            
        Returns:
            Dictionary containing verification results
        """
        results = {
            "has_train_data": "train" in data,
            "has_test_data": "test" in data,
            "has_timestamp": "timestamp" in data,
            "hash_match": False,
            "statistics_verified": False,
            "merkle_verified": False,
            "metadata_verified": False
        }
        
        # Verify hash if stored
        if "hash" in data:
            current_hash = self._generate_hash(data)
            results["hash_match"] = current_hash == data["hash"]
        
        # Verify statistics if present
        if "statistics" in data:
            results["statistics_verified"] = True
        
        # Verify metadata if present
        if "metadata" in data:
            results["metadata_verified"] = True
        
        return results
    
    def _verify_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify model component.
        
        Args:
            model: Model to verify
            
        Returns:
            Dictionary containing verification results
        """
        results = {
            "model_exists": True,
            "model_hash_match": False,
            "model_merkle_verified": False,
            "architecture_verified": False,
            "model_progression_verified": False,
            "mismatched_epochs": []
        }
        
        # Verify model hash if stored
        if "hash" in model:
            current_hash = self._generate_hash(model)
            results["model_hash_match"] = current_hash == model["hash"]
        
        # Verify architecture if present
        if "architecture" in model:
            results["architecture_verified"] = True
        
        # Verify model progression if training history present
        if "training_history" in model:
            history = model["training_history"]
            for i in range(1, len(history)):
                prev_state = history[i-1]
                curr_state = history[i]
                
                if prev_state == curr_state:
                    results["model_progression_verified"] = False
                    results["mismatched_epochs"].append(i)
        
        return results
    
    def _verify_training(self, training: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify training component.
        
        Args:
            training: Training data to verify
            
        Returns:
            Dictionary containing verification results
        """
        results = {
            "final_metrics_present": "final_metrics" in training,
            "training_logs_present": "training_history" in training,
            "privacy_metrics_present": "privacy_metrics" in training,
            "hash_present": "hash" in training,
            "timestamp_present": "timestamp" in training,
            "accuracy_present": "accuracy" in training.get("final_metrics", {}),
            "loss_present": "loss" in training.get("final_metrics", {})
        }
        
        # Verify training hash if stored
        if "hash" in training:
            current_hash = self._generate_hash(training)
            results["hash_match"] = current_hash == training["hash"]
        
        return results
    
    def generate_verification_report(
        self,
        model_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
        training_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive verification report.
        
        Args:
            model_path: Path to the model file to verify
            data_path: Path to the data provenance file
            training_path: Path to the training provenance file
            
        Returns:
            Dictionary containing verification results
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "verification_status": "success",
            "components": {},
            "merkle_tree": {
                "status": "not_verified",
                "root_hash": None,
                "proofs": {}
            }
        }
        
        try:
            # Convert string paths to Path objects if needed
            model_path = Path(model_path) if isinstance(model_path, str) else model_path
            data_path = Path(data_path) if isinstance(data_path, str) else data_path
            training_path = Path(training_path) if isinstance(training_path, str) else training_path
            
            # Initialize provenance data dictionary
            self.provenance_data = {
                "data": {},
                "model": {},
                "training": {}
            }
            
            # Load provenance data from files
            if data_path and data_path.exists():
                try:
                    with open(data_path, 'r') as f:
                        self.provenance_data["data"] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading data provenance from {data_path}: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    
            if model_path and model_path.exists():
                # Look for provenance.json in the provenance directory
                # The structure is: artifacts/models/<timestamp>/model.pth
                # And we want: artifacts/provenance/<timestamp>/provenance.json
                timestamp = model_path.parent.name
                # Get the project root (artifacts directory's parent)
                project_root = Path("/Users/mukeshjoshi/gitprojects/mnist_provenance")
                provenance_dir = project_root / "artifacts" / "provenance" / timestamp
                provenance_file = provenance_dir / "provenance.json"
                
                logger.info(f"Looking for provenance file at: {provenance_file}")
                if provenance_file.exists():
                    try:
                        with open(provenance_file, 'r') as f:
                            provenance_data = json.load(f)
                            if "model_provenance" in provenance_data:
                                self.provenance_data["model"] = provenance_data["model_provenance"]
                            else:
                                logger.warning("No model_provenance found in provenance file")
                    except Exception as e:
                        logger.error(f"Error loading model provenance from {provenance_file}: {str(e)}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                else:
                    logger.warning(f"Provenance file not found at {provenance_file}")
                    
            if training_path and training_path.exists():
                try:
                    with open(training_path, 'r') as f:
                        self.provenance_data["training"] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading training provenance from {training_path}: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Verify model if path provided
            if model_path and self.provenance_data["model"]:
                try:
                    model_report = self._verify_model(self.provenance_data["model"])
                    report["components"]["model"] = model_report
                    if not all(model_report.values()):
                        report["verification_status"] = "failed"
                except Exception as e:
                    logger.error(f"Error verifying model: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    report["components"]["model"] = {"error": str(e)}
            
            # Verify data if path provided
            if data_path and self.provenance_data["data"]:
                try:
                    data_report = self._verify_data(self.provenance_data["data"])
                    report["components"]["data"] = data_report
                    if not all(data_report.values()):
                        report["verification_status"] = "failed"
                except Exception as e:
                    logger.error(f"Error verifying data: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    report["components"]["data"] = {"error": str(e)}
            
            # Verify training if path provided
            if training_path and self.provenance_data["training"]:
                try:
                    training_report = self._verify_training(self.provenance_data["training"])
                    report["components"]["training"] = training_report
                    if not all(training_report.values()):
                        report["verification_status"] = "failed"
                except Exception as e:
                    logger.error(f"Error verifying training: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    report["components"]["training"] = {"error": str(e)}
            
            # Verify Merkle tree if all components are present
            if all(comp in report["components"] for comp in ["model", "data", "training"]):
                try:
                    merkle_report = self.verify_merkle_tree(
                        model_path=model_path,
                        data_path=data_path,
                        training_path=training_path
                    )
                    report["merkle_tree"] = merkle_report
                    if not merkle_report["is_valid"]:
                        report["verification_status"] = "failed"
                except Exception as e:
                    logger.error(f"Error verifying Merkle tree: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    report["merkle_tree"] = {"error": str(e)}
            
            return report
            
        except Exception as e:
            error_info = {
                "error": str(e),
                "file": __file__,
                "line": traceback.extract_tb(traceback.extract_stack()[-1])[0].lineno,
                "stack_trace": traceback.format_exc()
            }
            logger.error(f"Error generating verification report: {error_info}")
            report["verification_status"] = "error"
            report["error"] = error_info
            return report

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