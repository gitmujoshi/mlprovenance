import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from .hash_config import HashFactory

logger = logging.getLogger(__name__)

class ProvenanceData:
    """Class to store and manage ML provenance data."""
    
    def __init__(self, data: Dict[str, Any], timestamp: Optional[str] = None):
        """
        Initialize provenance data.
        
        Args:
            data: Dictionary containing provenance data
            timestamp: Optional timestamp for the data
        """
        self.data = data
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._hash = None
        logger.info(f"Initialized ProvenanceData with timestamp: {self.timestamp}")
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the data.
        
        Returns:
            Hex digest of the data hash
        """
        if self._hash is None:
            # Get hash function from factory
            hash_func = HashFactory.get_hash_function()
            
            # Serialize data to bytes
            data_bytes = json.dumps(self.data, sort_keys=True).encode()
            
            # Compute hash
            self._hash = hash_func(data_bytes)
            logger.info(f"Computed hash using {HashFactory.get_current_algorithm()}: {self._hash}")
        
        return self._hash
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert provenance data to dictionary.
        
        Returns:
            Dictionary containing data, timestamp, and hash
        """
        return {
            "data": self.data,
            "timestamp": self.timestamp,
            "hash": self.compute_hash()
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'ProvenanceData':
        """
        Create ProvenanceData instance from dictionary.
        
        Args:
            data_dict: Dictionary containing data, timestamp, and hash
            
        Returns:
            ProvenanceData instance
        """
        instance = cls(
            data=data_dict["data"],
            timestamp=data_dict.get("timestamp")
        )
        instance._hash = data_dict.get("hash")
        return instance 