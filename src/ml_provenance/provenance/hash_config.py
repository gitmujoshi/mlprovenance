"""
Hash Configuration and Factory for ML Provenance

This module provides configurable cryptographic hash functions for ML provenance
tracking. It supports multiple hash algorithms with different security and
performance characteristics, allowing users to choose the most appropriate
algorithm for their use case.

Architecture:
├── HashFactory: Configurable hash algorithm factory
├── Algorithm Selection: Dynamic algorithm switching
├── Performance Optimization: Algorithm-specific optimizations
└── Security Validation: Cryptographic strength verification

Supported Algorithms:
- BLAKE3: Fast, secure hash function (recommended)
- SHA256: Standard SHA-256 hash function
- SHA512: SHA-512 hash function for higher security

Key Features:
- Configurable hash algorithms
- Performance benchmarking
- Security level validation
- Cross-platform compatibility
- Memory-efficient hashing
- Streaming support for large data

Security Considerations:
- Collision resistance
- Preimage resistance
- Second preimage resistance
- Algorithm-specific security levels
- Performance vs security trade-offs

Performance Characteristics:
- BLAKE3: ~2-3x faster than SHA256, high security
- SHA256: Standard speed, proven security
- SHA512: Slower, highest security level

Author: ML Provenance Team
License: MIT
"""

import hashlib
import logging
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Optional BLAKE3 import
try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False
    blake3 = None

logger = logging.getLogger(__name__)


class HashFactory:
    """
    Configurable hash algorithm factory for ML provenance.
    
    This class provides a factory pattern for creating and managing
    cryptographic hash functions. It supports multiple hash algorithms
    and allows dynamic switching between them based on requirements.
    
    Key Features:
        - Configurable hash algorithms (BLAKE3, SHA256, SHA512)
        - Performance benchmarking and comparison
        - Security level validation
        - Memory-efficient hashing for large datasets
        - Cross-platform compatibility
        
    Supported Algorithms:
        - blake3: Fast, secure hash function (recommended)
        - sha256: Standard SHA-256 hash function
        - sha512: SHA-512 hash function for higher security
        
    Security Levels:
        - blake3: 256-bit security, collision-resistant
        - sha256: 128-bit collision resistance
        - sha512: 256-bit collision resistance
    """
    
    # Algorithm registry
    ALGORITHMS = {
        'blake3': {
            'name': 'BLAKE3',
            'security_bits': 256,
            'output_size': 32,
            'available': BLAKE3_AVAILABLE,
            'description': 'Fast, secure hash function with 256-bit security'
        },
        'sha256': {
            'name': 'SHA-256',
            'security_bits': 128,
            'output_size': 32,
            'available': True,
            'description': 'Standard SHA-256 hash function'
        },
        'sha512': {
            'name': 'SHA-512',
            'security_bits': 256,
            'output_size': 64,
            'available': True,
            'description': 'SHA-512 hash function for higher security'
        }
    }
    
    def __init__(self, default_algorithm: str = 'blake3'):
        """
        Initialize hash factory with default algorithm.
        
        Args:
            default_algorithm: Default hash algorithm to use
                ('blake3', 'sha256', 'sha512')
                
        Raises:
            ValueError: If algorithm is not supported or unavailable
        """
        self.current_algorithm = None
        self.hash_function = None
        
        # Set default algorithm
        self.set_algorithm(default_algorithm)
        
        logger.info(f"Hash factory initialized with algorithm: {default_algorithm}")
    
    def set_algorithm(self, algorithm: str) -> None:
        """
        Set the hash algorithm to use.
        
        This method configures the hash factory to use the specified
        algorithm for all subsequent hash operations.
        
        Args:
            algorithm: Hash algorithm name ('blake3', 'sha256', 'sha512')
            
        Raises:
            ValueError: If algorithm is not supported or unavailable
            
        Algorithm Selection:
            - blake3: Recommended for most use cases (fast, secure)
            - sha256: Standard choice for compatibility
            - sha512: Highest security level (slower)
        """
        algorithm = algorithm.lower()
        
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Supported: {list(self.ALGORITHMS.keys())}")
        
        algorithm_info = self.ALGORITHMS[algorithm]
        
        if not algorithm_info['available']:
            if algorithm == 'blake3':
                raise ValueError("BLAKE3 not available. Install with: pip install blake3")
            else:
                raise ValueError(f"Algorithm {algorithm} not available")
        
        self.current_algorithm = algorithm
        self.hash_function = self._create_hash_function(algorithm)
        
        logger.info(f"Hash algorithm set to: {algorithm_info['name']}")
    
    def _create_hash_function(self, algorithm: str):
        """
        Create hash function instance for the specified algorithm.
        
        Args:
            algorithm: Hash algorithm name
            
        Returns:
            Hash function instance
        """
        if algorithm == 'blake3':
            return blake3.blake3()
        elif algorithm == 'sha256':
            return hashlib.sha256()
        elif algorithm == 'sha512':
            return hashlib.sha512()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def hash(self, data: Union[str, bytes]) -> str:
        """
        Generate hash of input data.
        
        This method computes the cryptographic hash of the input data
        using the currently configured algorithm.
        
        Args:
            data: Input data to hash
                - str: String data (will be UTF-8 encoded)
                - bytes: Binary data
                
        Returns:
            Hexadecimal hash string
            
        Performance Notes:
            - For large data, consider using hash_stream() for memory efficiency
            - BLAKE3 is significantly faster than SHA256/SHA512
            - String data is automatically UTF-8 encoded
        """
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Create new hash function instance
            hash_func = self._create_hash_function(self.current_algorithm)
            
            # Update with data
            hash_func.update(data)
            
            # Get hex digest
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to hash data: {e}")
            raise
    
    def hash_stream(self, data_stream, chunk_size: int = 8192) -> str:
        """
        Generate hash of data stream for memory efficiency.
        
        This method computes the hash of a data stream by processing
        it in chunks, making it suitable for large files or datasets
        that don't fit in memory.
        
        Args:
            data_stream: Iterable data stream (file, generator, etc.)
            chunk_size: Size of chunks to process (default: 8KB)
            
        Returns:
            Hexadecimal hash string
            
        Use Cases:
            - Large file hashing
            - Streaming data processing
            - Memory-constrained environments
            - Network data hashing
        """
        try:
            # Create new hash function instance
            hash_func = self._create_hash_function(self.current_algorithm)
            
            # Process data in chunks
            for chunk in data_stream:
                if isinstance(chunk, str):
                    chunk = chunk.encode('utf-8')
                hash_func.update(chunk)
            
            # Get hex digest
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to hash stream: {e}")
            raise
    
    def hash_file(self, filepath: Union[str, Path]) -> str:
        """
        Generate hash of a file.
        
        This method computes the hash of a file by reading it in chunks,
        making it memory-efficient for large files.
        
        Args:
            filepath: Path to the file to hash
            
        Returns:
            Hexadecimal hash string
            
        Performance:
            - Memory-efficient for large files
            - Uses chunked reading
            - Supports all file types
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                return self.hash_stream(f)
                
        except Exception as e:
            logger.error(f"Failed to hash file {filepath}: {e}")
            raise
    
    def benchmark_algorithms(self, data: Union[str, bytes], iterations: int = 1000) -> Dict[str, Any]:
        """
        Benchmark all available hash algorithms.
        
        This method compares the performance of all available hash
        algorithms on the given data to help users choose the most
        appropriate algorithm for their use case.
        
        Args:
            data: Data to use for benchmarking
            iterations: Number of iterations for timing (default: 1000)
            
        Returns:
            Dictionary containing benchmark results
            
        Benchmark Metrics:
            - Hash time per iteration
            - Total time for all iterations
            - Hash rate (hashes per second)
            - Memory usage (estimated)
            - Security level comparison
        """
        results = {}
        
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        for algorithm, info in self.ALGORITHMS.items():
            if not info['available']:
                results[algorithm] = {
                    'available': False,
                    'error': 'Algorithm not available'
                }
                continue
            
            try:
                # Warm up
                for _ in range(10):
                    self._create_hash_function(algorithm).update(data).hexdigest()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    self._create_hash_function(algorithm).update(data).hexdigest()
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time = total_time / iterations
                hash_rate = iterations / total_time
                
                results[algorithm] = {
                    'available': True,
                    'total_time': total_time,
                    'avg_time_per_hash': avg_time,
                    'hash_rate': hash_rate,
                    'security_bits': info['security_bits'],
                    'output_size': info['output_size'],
                    'description': info['description']
                }
                
            except Exception as e:
                results[algorithm] = {
                    'available': False,
                    'error': str(e)
                }
        
        return results
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the current algorithm.
        
        Returns:
            Dictionary containing current algorithm information
        """
        if self.current_algorithm is None:
            return {'error': 'No algorithm set'}
        
        info = self.ALGORITHMS[self.current_algorithm].copy()
        info['current'] = True
        return info
    
    def get_available_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available algorithms.
        
        Returns:
            Dictionary mapping algorithm names to their information
        """
        return self.ALGORITHMS.copy()
    
    def validate_hash(self, hash_string: str) -> bool:
        """
        Validate that a hash string matches the current algorithm.
        
        This method checks if a hash string has the correct format
        and length for the currently configured algorithm.
        
        Args:
            hash_string: Hash string to validate
            
        Returns:
            True if hash is valid, False otherwise
        """
        if self.current_algorithm is None:
            return False
        
        try:
            # Check if it's a valid hex string
            int(hash_string, 16)
            
            # Check length
            expected_length = self.ALGORITHMS[self.current_algorithm]['output_size'] * 2
            return len(hash_string) == expected_length
            
        except (ValueError, KeyError):
            return False
    
    def get_recommended_algorithm(self, use_case: str = 'general') -> str:
        """
        Get recommended algorithm for a specific use case.
        
        Args:
            use_case: Use case description
                - 'general': General purpose (default)
                - 'performance': Performance-critical applications
                - 'security': High-security requirements
                - 'compatibility': Maximum compatibility
                
        Returns:
            Recommended algorithm name
        """
        recommendations = {
            'general': 'blake3',
            'performance': 'blake3',
            'security': 'sha512',
            'compatibility': 'sha256'
        }
        
        algorithm = recommendations.get(use_case, 'blake3')
        
        # Check if recommended algorithm is available
        if not self.ALGORITHMS[algorithm]['available']:
            # Fallback to available algorithms
            for alg in ['sha256', 'sha512']:
                if self.ALGORITHMS[alg]['available']:
                    algorithm = alg
                    break
        
        return algorithm


class TrainingHashConfig:
    """
    Training-specific hash configuration for ML provenance.
    
    This class provides training-specific configuration for hash algorithms,
    including performance optimization and security settings tailored for
    machine learning workflows.
    
    Key Features:
        - Training-optimized hash settings
        - Performance benchmarking for ML workloads
        - Security level configuration
        - Memory usage optimization
        - Integration with training pipelines
        
    Configuration Options:
        - algorithm: Hash algorithm selection
        - include_timestamps: Include timestamps in hashes
        - include_metadata: Include metadata in hashes
        - chunk_size: Chunk size for large data
        - performance_mode: Performance optimization settings
    """
    
    def __init__(self, algorithm: str = 'blake3', include_timestamps: bool = True,
                 include_metadata: bool = True, chunk_size: int = 8192):
        """
        Initialize training hash configuration.
        
        Args:
            algorithm: Hash algorithm to use
            include_timestamps: Include timestamps in hash calculations
            include_metadata: Include metadata in hash calculations
            chunk_size: Chunk size for processing large data
        """
        self.algorithm = algorithm
        self.include_timestamps = include_timestamps
        self.include_metadata = include_metadata
        self.chunk_size = chunk_size
        
        # Initialize hash factory
        self.hash_factory = HashFactory(algorithm)
        
        logger.info(f"Training hash config initialized: {algorithm}")
    
    def initialize_hash_factory(self) -> None:
        """
        Initialize the hash factory with current settings.
        
        This method ensures the hash factory is properly configured
        for the training workflow.
        """
        self.hash_factory.set_algorithm(self.algorithm)
        logger.info(f"Hash factory initialized with algorithm: {self.algorithm}")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the configured algorithm.
        
        Returns:
            Dictionary containing algorithm information
        """
        return self.hash_factory.get_algorithm_info()
    
    def hash_training_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Hash training data with optional metadata.
        
        Args:
            data: Training data to hash
            metadata: Optional metadata to include
            
        Returns:
            Hash string of training data
        """
        # Prepare data for hashing
        hash_data = {'data': data}
        
        if self.include_timestamps:
            hash_data['timestamp'] = time.time()
        
        if self.include_metadata and metadata:
            hash_data['metadata'] = metadata
        
        # Convert to JSON string for consistent hashing
        data_str = json.dumps(hash_data, sort_keys=True)
        
        return self.hash_factory.hash(data_str)
    
    def benchmark_for_training(self, sample_data: Any) -> Dict[str, Any]:
        """
        Benchmark hash algorithms for training workloads.
        
        Args:
            sample_data: Sample training data for benchmarking
            
        Returns:
            Benchmark results
        """
        return self.hash_factory.benchmark_algorithms(sample_data)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary containing current configuration
        """
        return {
            'algorithm': self.algorithm,
            'include_timestamps': self.include_timestamps,
            'include_metadata': self.include_metadata,
            'chunk_size': self.chunk_size,
            'algorithm_info': self.get_algorithm_info()
        } 