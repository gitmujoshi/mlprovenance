# Model Training Project with Provenance Tracking and Safety Features

## Overview

This project implements a machine learning model training pipeline with built-in provenance tracking and safety features. It includes a web application for testing the trained model with safety checks and content filtering.

## Project Structure

```
.
├── artifacts/           # Training artifacts and model checkpoints
├── data/               # Dataset storage
├── docs/               # Documentation
├── safety_features/    # Safety features implementation
├── scripts/            # Utility scripts
├── src/                # Source code
│   ├── ml_provenance/  # Provenance tracking implementation
│   └── safety_checks/  # Safety checks implementation
└── tests/              # Test files
```

## Key Components

### 1. Model Training Pipeline

The training pipeline includes:
- Model architecture definition
- Data loading and preprocessing
- Training loop with metrics tracking
- Model checkpointing
- Provenance tracking
- Safety feature integration

### 2. Provenance Tracking

The provenance tracking system captures:
- Data provenance (dataset size, statistics, hashes)
- Model architecture and parameters
- Training configuration and metrics
- System information
- Safety configuration

#### Provenance Data Structure

```json
{
    "version": "timestamp",
    "data_provenance": {
        "train": {
            "samples": "number of training examples",
            "statistics": {
                "mean": "mean value",
                "std": "standard deviation",
                "min": "minimum value",
                "max": "maximum value"
            }
        },
        "test": {
            "samples": "number of test examples",
            "statistics": {
                "mean": "mean value",
                "std": "standard deviation",
                "min": "minimum value",
                "max": "maximum value"
            }
        }
    },
    "model_provenance": {
        "architecture": {
            "name": "model name",
            "layers": [
                {
                    "name": "layer name",
                    "type": "layer type",
                    "parameters": "number of parameters"
                }
            ],
            "total_parameters": "total number of parameters"
        }
    },
    "training_provenance": {
        "config": {
            "epochs": "number of epochs",
            "batch_size": "batch size",
            "learning_rate": "learning rate"
        },
        "metrics": {
            "train_loss": "training loss",
            "val_loss": "validation loss",
            "train_acc": "training accuracy",
            "val_acc": "validation accuracy"
        }
    }
}
```

### 3. Model Validation with Merkle Trees

The project implements a Merkle tree-based validation system to ensure model integrity and provenance. This allows users to verify that the model they're using is exactly the same as the one that was trained and hasn't been tampered with.

#### How to Validate the Model

1. **Get the Model Hash**
   ```python
   from ml_provenance.provenance.tracker import Tracker
   
   # Initialize tracker
   tracker = Tracker()
   
   # Get model hash
   model_hash = tracker.get_model_hash("path/to/model")
   ```

2. **Verify Against Provenance Data**
   ```python
   # Load provenance data
   provenance_data = tracker.load_provenance()
   
   # Verify model hash
   is_valid = tracker.verify_model_hash(model_hash, provenance_data)
   if is_valid:
       print("Model is valid and matches the training provenance")
   else:
       print("Model validation failed - possible tampering detected")
   ```

3. **Verify Individual Components**
   ```python
   # Verify specific components
   components = {
       "model_weights": "hash_of_weights",
       "training_config": "hash_of_config",
       "safety_config": "hash_of_safety"
   }
   
   for component, hash_value in components.items():
       is_valid = tracker.verify_component_hash(component, hash_value, provenance_data)
       print(f"{component}: {'Valid' if is_valid else 'Invalid'}")
   ```

#### Merkle Tree Structure

The Merkle tree is constructed as follows:

```
                    [Root Hash]
                    /         \
            [Training Hash]  [Config Hash]
            /     |     \    /     |     \
    [Weights] [Metrics] [Data] [Safety] [System]
```

1. **Leaf Nodes**
   - Model weights hash
   - Training configuration hash
   - Safety configuration hash
   - Dataset statistics hash
   - System information hash

2. **Intermediate Nodes**
   - Combined hashes of related components
   - Training-related hashes
   - Configuration-related hashes

3. **Root Hash**
   - Final hash representing the entire model state
   - Stored in the provenance data
   - Used for quick validation

#### Validation Scenarios

1. **Initial Model Validation**
   ```python
   # First-time validation of a new model
   def validate_new_model(model_path, provenance_path):
       validator = MerkleValidator()
       result = validator.validate_new_model(model_path, provenance_path)
       
       if result.is_valid:
           print("Model successfully validated and registered")
           print(f"Model ID: {result.model_id}")
           print(f"Validation Timestamp: {result.timestamp}")
       else:
           print("Validation failed:")
           for error in result.errors:
               print(f"- {error}")
   ```

2. **Incremental Update Validation**
   ```python
   # Validate model after updates
   def validate_model_update(model_path, provenance_path, previous_hash):
       validator = MerkleValidator()
       result = validator.validate_update(model_path, previous_hash)
       
       if result.is_valid:
           print("Update validated successfully")
           print(f"Changed components: {result.changed_components}")
       else:
           print("Update validation failed")
           print(f"Unexpected changes: {result.unexpected_changes}")
   ```

3. **Component-Specific Validation**
   ```python
   # Validate specific model components
   def validate_components(model_path, components):
       validator = MerkleValidator()
       results = {}
       
       for component in components:
           result = validator.validate_component(model_path, component)
           results[component] = result
           
           if result.is_valid:
               print(f"{component}: Valid")
           else:
               print(f"{component}: Invalid")
               print(f"Reason: {result.reason}")
               
       return results
   ```

#### Handling Validation Failures

1. **Common Failure Scenarios**

   a. **Hash Mismatch**
   ```python
   try:
       validator.validate_model(model_path, provenance_path)
   except HashMismatchError as e:
       print(f"Hash mismatch detected: {e}")
       print(f"Expected: {e.expected_hash}")
       print(f"Actual: {e.actual_hash}")
       # Log the mismatch and notify administrators
   ```

   b. **Missing Components**
   ```python
   try:
       validator.validate_model(model_path, provenance_path)
   except MissingComponentError as e:
       print(f"Missing component: {e.component}")
       print(f"Required by: {e.required_by}")
       # Attempt to recover or fetch missing component
   ```

   c. **Corrupted Data**
   ```python
   try:
       validator.validate_model(model_path, provenance_path)
   except CorruptedDataError as e:
       print(f"Data corruption detected in: {e.component}")
       print(f"Corruption type: {e.corruption_type}")
       # Attempt data recovery or re-download
   ```

2. **Recovery Procedures**

   a. **Automatic Recovery**
   ```python
   def attempt_recovery(model_path, error):
       recovery = ModelRecovery()
       
       if isinstance(error, HashMismatchError):
           # Attempt to repair corrupted files
           recovery.repair_corrupted_files(model_path)
       elif isinstance(error, MissingComponentError):
           # Download missing components
           recovery.download_missing_components(error.component)
       elif isinstance(error, CorruptedDataError):
           # Attempt data recovery
           recovery.recover_corrupted_data(error.component)
   ```

   b. **Manual Recovery**
   ```python
   def manual_recovery_guide(error):
       print("Manual Recovery Required")
       print("1. Stop all model operations")
       print("2. Backup current model state")
       print("3. Follow these steps:")
       
       if isinstance(error, HashMismatchError):
           print("   a. Download original model from trusted source")
           print("   b. Verify download integrity")
           print("   c. Replace corrupted files")
       elif isinstance(error, MissingComponentError):
           print("   a. Identify missing components")
           print("   b. Obtain components from backup")
           print("   c. Verify component integrity")
   ```

3. **Validation Logging**

   ```python
   class ValidationLogger:
       def __init__(self):
           self.logger = logging.getLogger('model_validation')
           
       def log_validation_attempt(self, model_path, result):
           self.logger.info(f"Validation attempt for {model_path}")
           self.logger.info(f"Result: {'Success' if result.is_valid else 'Failure'}")
           
           if not result.is_valid:
               self.logger.error("Validation errors:")
               for error in result.errors:
                   self.logger.error(f"- {error}")
                   
       def log_recovery_attempt(self, error, success):
           self.logger.info(f"Recovery attempt for {error.__class__.__name__}")
           self.logger.info(f"Recovery {'successful' if success else 'failed'}")
   ```

#### Security Considerations

1. **Hash Function Security**
   - Use SHA-256 or stronger hash functions
   - Regularly update hash functions
   - Implement hash function fallbacks

2. **Provenance Data Protection**
   - Encrypt provenance data at rest
   - Use secure channels for transmission
   - Implement access controls

3. **Validation Process Security**
   - Validate in isolated environment
   - Implement rate limiting
   - Log all validation attempts

### 4. Safety Features

The safety system includes:
- Content filtering
- Age rating checks
- Input/output length limits
- Sensitive topic detection
- Content warning requirements

#### Safety Configuration

```python
safety_config = SafetyConfig(
    min_age_rating=AgeRating.TEEN,
    content_filters=["violence", "explicit", "offensive"],
    max_input_length=512,
    max_output_length=100,
    block_sensitive_topics=True,
    require_content_warning=True
)
```

### 5. Web Application

The web application provides:
- Model information display
- Text generation interface
- Safety check results
- Example prompts for testing

#### Features

1. **Model Information Display**
   - Model path and last modified date
   - Training configuration
   - Dataset size
   - Safety features configuration

2. **Text Generation**
   - Input prompt field
   - Generation controls
   - Output display
   - Safety check results

3. **Safety Checks**
   - Input validation
   - Output filtering
   - Age rating verification
   - Content warning application

## Setup and Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up training environment:
```bash
./setup_training_env.sh
```

## Training the Model

1. Run the training script:
```bash
python safety_features/scripts/train_gpt2_with_safety.py \
    --epochs 3 \
    --batch-size 8 \
    --min-age-rating TEEN \
    --max-input-length 512 \
    --content-filters violence explicit offensive \
    --block-sensitive-topics \
    --require-content-warning
```

2. Monitor training progress:
- Training metrics are logged to the console
- Provenance data is saved in the artifacts directory
- Model checkpoints are saved automatically

## Running the Web Application

1. Start the Flask application:
```bash
python safety_features/scripts/run_app.py
```

2. Access the web interface:
- Open http://localhost:5000 in your browser
- View model information
- Test the model with different prompts
- Check safety feature results

## Testing Safety Features

The web interface includes example prompts to test different safety features:

1. **Safe Content**
   - "Tell me a fun fact about space"
   - "What are the benefits of reading books?"
   - "How do plants make food?"

2. **Content Warning Triggers**
   - "Write a story about a war between two kingdoms"
   - "Describe a natural disaster"

3. **Length Limit Tests**
   - Very long input prompts
   - Extended generation requests

4. **Sensitive Topics**
   - Political content
   - Controversial subjects

5. **Inappropriate Content**
   - Explicit material
   - Offensive language

## Troubleshooting

Common issues and solutions:

1. **Port Already in Use**
   - On macOS, disable AirPlay Receiver
   - Use a different port: `python run_app.py --port 5001`

2. **Model Loading Issues**
   - Check artifacts directory for model files
   - Verify model path in configuration
   - Ensure all dependencies are installed

3. **Safety Check Failures**
   - Review safety configuration
   - Check input/output length limits
   - Verify content filter settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

[Add your license information here]

## Contact

[Add your contact information here] 