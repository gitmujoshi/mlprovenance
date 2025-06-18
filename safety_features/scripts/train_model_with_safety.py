#!/usr/bin/env python3
"""
Training script with safety features and configurable hash algorithms.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from safety_features.frameworks.pytorch import PyTorchSafetyWrapper
from safety_features.frameworks.tensorflow import TensorFlowSafetyWrapper
from safety_features.frameworks.jax import JAXSafetyWrapper
from ml_provenance.provenance.hash_config import TrainingHashConfig

def load_config(config_path: str) -> dict:
    """Load training configuration from file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def initialize_hash_config(config: dict):
    """Initialize hash configuration from training config."""
    hash_config = TrainingHashConfig(config)
    hash_config.initialize_hash_factory()
    
    logging.info(f"Using hash algorithm: {hash_config.get_current_algorithm()}")
    logging.info(f"Algorithm info: {hash_config.get_algorithm_info()}")
    
    return hash_config

def train_with_safety(model_name: str, config_path: str, framework: str = "pytorch"):
    """Train model with safety features and configurable hash algorithm."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize hash configuration
    hash_config = initialize_hash_config(config)
    
    # Framework-specific training
    if framework == "pytorch":
        from safety_features.frameworks.pytorch import PyTorchSafetyTrainer
        trainer = PyTorchSafetyTrainer(model_name, config)
    elif framework == "tensorflow":
        from safety_features.frameworks.tensorflow import TensorFlowSafetyTrainer
        trainer = TensorFlowSafetyTrainer(model_name, config)
    elif framework == "jax":
        from safety_features.frameworks.jax import JAXSafetyTrainer
        trainer = JAXSafetyTrainer(model_name, config)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    # Train the model
    model, metrics = trainer.train()
    
    logging.info(f"Training completed with hash algorithm: {hash_config.get_current_algorithm()}")
    logging.info(f"Final metrics: {metrics}")
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Train model with safety features")
    parser.add_argument("--model_name", required=True, help="Name of the model to train")
    parser.add_argument("--config_path", required=True, help="Path to training configuration file")
    parser.add_argument("--framework", default="pytorch", choices=["pytorch", "tensorflow", "jax"], 
                       help="ML framework to use")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        model, metrics = train_with_safety(args.model_name, args.config_path, args.framework)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 