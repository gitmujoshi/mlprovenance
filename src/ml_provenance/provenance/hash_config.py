import hashlib
import blake3
from typing import Callable, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HashConfig:
    """Configuration for hash functions used in provenance tracking."""
    
    def __init__(self, hash_algorithm: str = "sha256"):
        """
        Initialize hash configuration.
        
        Args:
            hash_algorithm: The hash algorithm to use. Supported values:
                - "sha256" (default)
                - "blake3"
                - "sha512"
        """
        self.hash_algorithm = hash_algorithm.lower()
        self._validate_algorithm()
        
    def _validate_algorithm(self):
        """Validate the selected hash algorithm."""
        supported_algorithms = ["sha256", "blake3", "sha512"]
        if self.hash_algorithm not in supported_algorithms:
            raise ValueError(
                f"Unsupported hash algorithm: {self.hash_algorithm}. "
                f"Supported algorithms are: {', '.join(supported_algorithms)}"
            )
    
    def get_hash_function(self) -> Callable[[bytes], str]:
        """
        Get the configured hash function.
        
        Returns:
            A function that takes bytes and returns a hex digest string.
        """
        if self.hash_algorithm == "sha256":
            return lambda data: hashlib.sha256(data).hexdigest()
        elif self.hash_algorithm == "blake3":
            return lambda data: blake3.blake3(data).hexdigest()
        elif self.hash_algorithm == "sha512":
            return lambda data: hashlib.sha512(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

class HashFactory:
    """Factory for creating hash functions with consistent configuration."""
    
    _instance = None
    _config: Optional[HashConfig] = None
    
    @classmethod
    def initialize(cls, hash_algorithm: str = "sha256"):
        """
        Initialize the hash factory with a specific algorithm.
        
        Args:
            hash_algorithm: The hash algorithm to use.
        """
        cls._config = HashConfig(hash_algorithm)
        logger.info(f"Hash factory initialized with algorithm: {hash_algorithm}")
    
    @classmethod
    def get_hash_function(cls) -> Callable[[bytes], str]:
        """
        Get the configured hash function.
        
        Returns:
            A function that takes bytes and returns a hex digest string.
        """
        if cls._config is None:
            cls.initialize()  # Initialize with default SHA-256
        return cls._config.get_hash_function()
    
    @classmethod
    def get_current_algorithm(cls) -> str:
        """
        Get the currently configured hash algorithm.
        
        Returns:
            The name of the current hash algorithm.
        """
        if cls._config is None:
            cls.initialize()
        return cls._config.hash_algorithm 