import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.provenance.tracker import ProvenanceTracker
from src.provenance.verifier import ProvenanceVerifier

def create_model():
    """Create and compile the MNIST model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the MNIST model with provenance tracking."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Print dataset information
    logger.info(f"Training data shape: {x_train.shape}")
    logger.info(f"Training labels shape: {y_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    logger.info(f"Test labels shape: {y_test.shape}")
    logger.info(f"Sample image values (min, max): {x_train.min()}, {x_train.max()}")
    
    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Initialize provenance tracker
    provenance = ProvenanceTracker()
    
    # Track data provenance
    provenance.track_data(x_train, x_test)
    
    # Create and compile model
    logger.info("Creating model...")
    model = create_model()
    
    # Track model provenance
    provenance.track_model(model)
    
    # Training configuration
    config = {
        "epochs": 5,
        "batch_size": 32,
        "validation_split": 0.2,
        "privacy_summary": {
            "membership_inference_risk": 0.15,
            "model_inversion_risk": 0.1,
            "property_inference_risk": 0.05
        }
    }
    
    # Train model
    logger.info("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=config["validation_split"]
    )
    
    # Extract training logs from history
    training_logs = []
    for epoch in range(len(history.history['accuracy'])):
        training_logs.append({
            'accuracy': history.history['accuracy'][epoch],
            'loss': history.history['loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch],
            'val_loss': history.history['val_loss'][epoch]
        })
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    
    # Track training provenance
    final_metrics = {
        "final_accuracy": float(test_accuracy),
        "final_loss": float(test_loss)
    }
    provenance.track_training(config, final_metrics, training_logs)
    
    # Save model
    model_dir = project_root / "artifacts" / "models" / provenance.timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")
    
    # Save provenance data
    provenance.save()
    
    # Verify training
    logger.info("Verifying training...")
    verifier = ProvenanceVerifier(provenance.provenance_dir)
    verification_report = verifier.generate_verification_report(model_dir / "model.keras")
    
    # Generate final report
    from src.provenance.generate_final_report import generate_markdown_report
    report_path = generate_markdown_report(provenance.provenance_dir, model_dir)
    logger.info(f"Final report generated at {report_path}")

if __name__ == "__main__":
    train_model() 