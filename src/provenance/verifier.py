import json
import logging
from pathlib import Path
import tensorflow as tf
import hashlib
import numpy as np
import base64

class ProvenanceVerifier:
    def __init__(self, provenance_dir):
        self.provenance_dir = Path(provenance_dir)
        self.logger = logging.getLogger(__name__)
        
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
    
    def generate_verification_report(self, model_path):
        """Generate a verification report for the training run."""
        self.logger.info("Generating verification report...")
        
        # Load provenance data
        with open(self.provenance_dir / "data.json", "r") as f:
            data = json.load(f)
        
        # Initialize verification results
        verification_results = {
            "data_verification": self._verify_data(data),
            "model_verification": self._verify_model(data, model_path),
            "training_verification": self._verify_training(data),
            "hash_verification": self._verify_hashes(data, model_path)
        }
        
        # Determine overall status
        overall_status = all(
            all(check for check in results.values())
            for results in verification_results.values()
        )
        
        # Create verification report
        report = {
            "provenance_version": data["version"],
            "overall_status": overall_status,
            "verification_results": verification_results,
            "verification_timestamp": tf.timestamp().numpy().astype(str)
        }
        
        # Save verification report
        with open(self.provenance_dir / "verification.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _verify_data(self, data):
        """Verify data provenance."""
        self.logger.info("Verifying data provenance...")
        
        results = {
            "train": "train" in data["data_provenance"],
            "test": "test" in data["data_provenance"],
            "timestamp": "version" in data,
            "train_hash": "hash" in data["data_provenance"]["train"],
            "test_hash": "hash" in data["data_provenance"]["test"]
        }
        
        return results
    
    def _verify_model(self, data, model_path):
        """Verify model provenance."""
        self.logger.info(f"Verifying model at {model_path}...")
        
        model = tf.keras.models.load_model(model_path)
        model_config = model.get_config()
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
        
        # Generate hash using only the numpy arrays
        weights_hash = self._generate_hash([w[2] for w in weights])
        
        stored_model_hash = data["model_provenance"]["hashes"]["architecture"]
        stored_weights_hash = data["model_provenance"]["hashes"]["weights"]
        
        # Check that weights have changed during training
        weights_changed = weights_hash != stored_weights_hash
        
        results = {
            "model_hash_match": model_hash == stored_model_hash,
            "weights_changed": weights_changed,
            "model_exists": Path(model_path).exists(),
            "architecture_verification": self._verify_model_architecture(model, data)
        }
        
        return results
    
    def _verify_model_architecture(self, model, data):
        """Verify model architecture details."""
        stored_config = data["model_provenance"]["architecture"]
        current_config = model.get_config()
        
        return {
            "layer_count_match": len(stored_config["layers"]) == len(current_config["layers"]),
            "parameter_count_match": model.count_params() == data["model_provenance"]["parameters"]["trainable_params"],
            "optimizer_match": model.optimizer.__class__.__name__ == data["model_provenance"]["parameters"]["optimizer"]["name"]
        }
    
    def _verify_training(self, data):
        """Verify training provenance."""
        self.logger.info("Verifying training process...")
        
        results = {
            "test_accuracy_present": "final_accuracy" in data["training_provenance"]["final_metrics"],
            "test_loss_present": "final_loss" in data["training_provenance"]["final_metrics"],
            "privacy_metrics_present": "privacy_summary" in data["training_provenance"]["config"],
            "training_hash_present": "hash" in data["training_provenance"]
        }
        
        return results
    
    def _verify_hashes(self, data, model_path):
        """Verify all hashes in the provenance data."""
        self.logger.info("Verifying provenance hashes...")
        
        # Load model for hash verification
        model = tf.keras.models.load_model(model_path)
        
        # Generate current hashes
        current_hashes = {
            "model_architecture": self._generate_hash(model.get_config()),
            "model_weights": self._generate_hash([w.numpy().astype(np.float32) for w in model.weights]),
            "training": self._generate_hash({
                "config": data["training_provenance"]["config"],
                "metrics": data["training_provenance"]["final_metrics"]
            })
        }
        
        # Compare with stored hashes
        results = {
            "model_architecture_hash_match": current_hashes["model_architecture"] == data["hashes"]["model_architecture"],
            "model_weights_changed": current_hashes["model_weights"] != data["hashes"]["model_weights"],
            "training_hash_match": current_hashes["training"] == data["hashes"]["training"],
            "overall_hash_present": "overall" in data["hashes"]
        }
        
        return results 