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
                - "sha256" (default) - SHA-256, widely used and secure
                - "blake3" - BLAKE3, very fast and secure
                - "sha512" - SHA-512, longer hash for higher security
                - "sha1" - SHA-1, legacy support (not recommended for security)
                - "md5" - MD5, legacy support (not recommended for security)
        """
        self.hash_algorithm = hash_algorithm.lower()
        self._validate_algorithm()
        
    def _validate_algorithm(self):
        """Validate the selected hash algorithm."""
        supported_algorithms = ["sha256", "blake3", "sha512", "sha1", "md5"]
        if self.hash_algorithm not in supported_algorithms:
            raise ValueError(
                f"Unsupported hash algorithm: {self.hash_algorithm}. "
                f"Supported algorithms are: {', '.join(supported_algorithms)}"
            )
        
        # Warn about deprecated algorithms
        if self.hash_algorithm in ["sha1", "md5"]:
            logger.warning(
                f"Hash algorithm '{self.hash_algorithm}' is deprecated and not recommended "
                "for security-critical applications. Consider using 'sha256' or 'blake3'."
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
        elif self.hash_algorithm == "sha1":
            return lambda data: hashlib.sha1(data).hexdigest()
        elif self.hash_algorithm == "md5":
            return lambda data: hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the configured hash algorithm.
        
        Returns:
            Dictionary containing algorithm information
        """
        algorithm_info = {
            "sha256": {
                "name": "SHA-256",
                "digest_size": 32,
                "security_level": "high",
                "speed": "medium",
                "recommended": True
            },
            "blake3": {
                "name": "BLAKE3",
                "digest_size": 32,
                "security_level": "high",
                "speed": "very_fast",
                "recommended": True
            },
            "sha512": {
                "name": "SHA-512",
                "digest_size": 64,
                "security_level": "very_high",
                "speed": "slow",
                "recommended": True
            },
            "sha1": {
                "name": "SHA-1",
                "digest_size": 20,
                "security_level": "broken",
                "speed": "fast",
                "recommended": False
            },
            "md5": {
                "name": "MD5",
                "digest_size": 16,
                "security_level": "broken",
                "speed": "very_fast",
                "recommended": False
            }
        }
        
        return algorithm_info.get(self.hash_algorithm, {})

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
        
        # Log algorithm information
        info = cls._config.get_algorithm_info()
        logger.info(f"Algorithm: {info.get('name', hash_algorithm)}")
        logger.info(f"Digest size: {info.get('digest_size', 'unknown')} bytes")
        logger.info(f"Security level: {info.get('security_level', 'unknown')}")
        logger.info(f"Speed: {info.get('speed', 'unknown')}")
        logger.info(f"Recommended: {info.get('recommended', 'unknown')}")
    
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
    
    @classmethod
    def get_algorithm_info(cls) -> Dict[str, Any]:
        """
        Get information about the currently configured hash algorithm.
        
        Returns:
            Dictionary containing algorithm information
        """
        if cls._config is None:
            cls.initialize()
        return cls._config.get_algorithm_info()
    
    @classmethod
    def get_supported_algorithms(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all supported hash algorithms.
        
        Returns:
            Dictionary containing information about all supported algorithms
        """
        algorithms = {}
        for algo in ["sha256", "blake3", "sha512", "sha1", "md5"]:
            config = HashConfig(algo)
            algorithms[algo] = config.get_algorithm_info()
        return algorithms

class TrainingHashConfig:
    """Configuration for hash functions in training scenarios."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training hash configuration from training config.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.hash_algorithm = self._get_hash_algorithm_from_config()
        self.hash_config = HashConfig(self.hash_algorithm)
        
    def _get_hash_algorithm_from_config(self) -> str:
        """
        Extract hash algorithm from training configuration.
        
        Returns:
            Hash algorithm name
        """
        # Check for hash algorithm in various config locations
        hash_algo = (
            self.config.get("hash_algorithm") or
            self.config.get("provenance", {}).get("hash_algorithm") or
            self.config.get("safety", {}).get("hash_algorithm") or
            "sha256"  # default
        )
        
        logger.info(f"Using hash algorithm from config: {hash_algo}")
        return hash_algo
    
    def initialize_hash_factory(self):
        """Initialize the hash factory with the configured algorithm."""
        HashFactory.initialize(self.hash_algorithm)
    
    def get_hash_function(self) -> Callable[[bytes], str]:
        """Get the configured hash function."""
        return self.hash_config.get_hash_function()
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the configured hash algorithm."""
        return self.hash_config.get_algorithm_info() 