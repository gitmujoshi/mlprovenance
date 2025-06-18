#!/usr/bin/env python3
"""
Test script to demonstrate different hash algorithms and their performance.
"""

import time
import hashlib
import blake3
import json
from typing import Dict, Any
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml_provenance.provenance.hash_config import HashFactory, HashConfig

def test_hash_algorithms():
    """Test different hash algorithms and their performance."""
    
    # Sample data for testing
    sample_data = {
        "model_weights": [0.1, 0.2, 0.3, 0.4, 0.5],
        "training_config": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "dataset_info": {
            "name": "MNIST",
            "samples": 60000,
            "features": 784
        }
    }
    
    # Convert to bytes for hashing
    data_bytes = json.dumps(sample_data, sort_keys=True).encode()
    
    print("Hash Algorithm Performance Test")
    print("=" * 50)
    print(f"Data size: {len(data_bytes)} bytes")
    print()
    
    # Test each algorithm
    algorithms = ["sha256", "blake3", "sha512", "sha1", "md5"]
    results = {}
    
    for algo in algorithms:
        print(f"Testing {algo.upper()}...")
        
        # Initialize hash factory
        HashFactory.initialize(algo)
        hash_func = HashFactory.get_hash_function()
        info = HashFactory.get_algorithm_info()
        
        # Performance test
        start_time = time.time()
        for _ in range(1000):  # 1000 iterations for performance measurement
            hash_result = hash_func(data_bytes)
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        avg_time = total_time / 1000
        hashes_per_second = 1000 / total_time
        
        results[algo] = {
            "hash": hash_result,
            "total_time": total_time,
            "avg_time": avg_time,
            "hashes_per_second": hashes_per_second,
            "info": info
        }
        
        print(f"  Hash: {hash_result[:16]}...")
        print(f"  Time: {total_time:.4f}s for 1000 hashes")
        print(f"  Avg: {avg_time:.6f}s per hash")
        print(f"  Speed: {hashes_per_second:.0f} hashes/second")
        print(f"  Security: {info.get('security_level', 'unknown')}")
        print(f"  Recommended: {info.get('recommended', 'unknown')}")
        print()
    
    # Print comparison table
    print("Performance Comparison")
    print("=" * 50)
    print(f"{'Algorithm':<10} {'Speed (H/s)':<12} {'Security':<10} {'Recommended':<12}")
    print("-" * 50)
    
    for algo in algorithms:
        result = results[algo]
        speed = f"{result['hashes_per_second']:.0f}"
        security = result['info'].get('security_level', 'unknown')
        recommended = "Yes" if result['info'].get('recommended', False) else "No"
        print(f"{algo.upper():<10} {speed:<12} {security:<10} {recommended:<12}")
    
    print()
    
    # Recommendations
    print("Recommendations:")
    print("- SHA-256: Best for general use, good balance of security and performance")
    print("- BLAKE3: Best for high-performance applications")
    print("- SHA-512: Best for maximum security requirements")
    print("- SHA-1/MD5: Avoid for security-critical applications")

def test_configurable_hashing():
    """Test configurable hashing from training configuration."""
    
    print("\nConfigurable Hashing Test")
    print("=" * 50)
    
    # Sample training configurations
    configs = [
        {
            "name": "High Performance",
            "config": {
                "hash_algorithm": "blake3",
                "model": {"type": "transformer"},
                "training": {"epochs": 10}
            }
        },
        {
            "name": "Maximum Security",
            "config": {
                "hash_algorithm": "sha512",
                "model": {"type": "transformer"},
                "training": {"epochs": 10}
            }
        },
        {
            "name": "Standard",
            "config": {
                "hash_algorithm": "sha256",
                "model": {"type": "transformer"},
                "training": {"epochs": 10}
            }
        }
    ]
    
    sample_data = {"test": "data", "value": 123}
    data_bytes = json.dumps(sample_data, sort_keys=True).encode()
    
    for test_config in configs:
        print(f"\n{test_config['name']} Configuration:")
        
        # Initialize hash config from training config
        from src.ml_provenance.provenance.hash_config import TrainingHashConfig
        hash_config = TrainingHashConfig(test_config['config'])
        hash_config.initialize_hash_factory()
        
        # Get hash function and compute hash
        hash_func = hash_config.get_hash_function()
        hash_result = hash_func(data_bytes)
        info = hash_config.get_algorithm_info()
        
        print(f"  Algorithm: {hash_config.get_current_algorithm()}")
        print(f"  Hash: {hash_result}")
        print(f"  Security: {info.get('security_level', 'unknown')}")
        print(f"  Speed: {info.get('speed', 'unknown')}")

if __name__ == "__main__":
    test_hash_algorithms()
    test_configurable_hashing() 