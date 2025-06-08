from typing import Dict, Any, Optional
import hashlib
import json
from datetime import datetime

class ProvenanceData:
    """Base class for tracking ML provenance data with consistent structure."""
    
    def __init__(self):
        self.data = {
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data": {
                "hash": None,
                "metadata": {
                    "train_samples": None,
                    "test_samples": None,
                    "dataset": None,
                    "timestamp": None
                }
            },
            "model": {
                "hash": None,
                "metadata": {
                    "architecture": None,
                    "parameters": None,
                    "timestamp": None
                }
            },
            "training": {
                "hash": None,
                "metadata": {
                    "epochs": None,
                    "batch_size": None,
                    "learning_rate": None,
                    "final_metrics": None,
                    "timestamp": None
                }
            },
            "privacy": {
                "hash": None,
                "metadata": {
                    "target_epsilon": None,
                    "target_delta": None,
                    "noise_multiplier": None,
                    "achieved_epsilon": None,
                    "achieved_delta": None,
                    "timestamp": None
                }
            }
        }
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of the given data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def update_data(self, data_info: Dict[str, Any]) -> None:
        """Update data provenance information."""
        self.data["data"]["metadata"].update(data_info)
        self.data["data"]["hash"] = self._compute_hash(self.data["data"]["metadata"])
    
    def update_model(self, model_info: Dict[str, Any]) -> None:
        """Update model provenance information."""
        self.data["model"]["metadata"].update(model_info)
        self.data["model"]["hash"] = self._compute_hash(self.data["model"]["metadata"])
    
    def update_training(self, training_info: Dict[str, Any]) -> None:
        """Update training provenance information."""
        self.data["training"]["metadata"].update(training_info)
        self.data["training"]["hash"] = self._compute_hash(self.data["training"]["metadata"])
    
    def update_privacy(self, privacy_info: Dict[str, Any]) -> None:
        """Update privacy provenance information."""
        self.data["privacy"]["metadata"].update(privacy_info)
        self.data["privacy"]["hash"] = self._compute_hash(self.data["privacy"]["metadata"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance data to dictionary."""
        return self.data
    
    def to_json(self) -> str:
        """Convert provenance data to JSON string."""
        return json.dumps(self.data, indent=2)
    
    def save(self, filepath: str) -> None:
        """Save provenance data to file."""
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProvenanceData':
        """Load provenance data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        instance = cls()
        instance.data = data
        return instance 