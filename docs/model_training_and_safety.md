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

#### Safety Verification with Merkle Trees

The Merkle tree includes a dedicated `safety_metrics` section that allows users to verify that the model meets all safety requirements. This is crucial for ensuring that deployed applications maintain the safety standards established during training.

1. **Safety Metrics Structure**
   ```json
   {
     "safety_metrics": {
       "content_warnings": "number of content warnings generated",
       "failed_checks": "number of failed safety checks",
       "passed_checks": "number of passed safety checks",
       "total_checks": "total number of safety checks performed"
     }
   }
   ```

2. **Verifying Safety Requirements**
   ```python
   from ml_provenance.provenance.tracker import Tracker
   from ml_provenance.safety.verifier import SafetyVerifier
   
   def verify_model_safety(model_path, provenance_path):
       # Initialize components
       tracker = Tracker()
       verifier = SafetyVerifier()
       
       # Load provenance data
       provenance_data = tracker.load_provenance(provenance_path)
       
       # Get safety metrics from Merkle tree
       safety_metrics = tracker.get_safety_metrics()
       
       # Verify safety requirements
       safety_report = verifier.verify_safety_requirements(
           safety_metrics,
           min_pass_rate=0.95,  # Minimum required pass rate
           max_content_warnings=100,  # Maximum allowed content warnings
           required_checks=['age_rating', 'content_filter', 'sensitive_topics']
       )
       
       return safety_report
   ```

3. **Safety Verification Report**
   ```python
   def generate_safety_report(safety_metrics):
       report = {
           "overall_status": "PASS" if safety_metrics["passed_checks"] / safety_metrics["total_checks"] >= 0.95 else "FAIL",
           "metrics": {
               "pass_rate": f"{safety_metrics['passed_checks'] / safety_metrics['total_checks']:.2%}",
               "content_warnings": safety_metrics["content_warnings"],
               "failed_checks": safety_metrics["failed_checks"]
           },
           "requirements": {
               "min_pass_rate": "95%",
               "max_content_warnings": "100",
               "required_checks": ["age_rating", "content_filter", "sensitive_topics"]
           }
       }
       return report
   ```

4. **Continuous Safety Monitoring**
   ```python
   class SafetyMonitor:
       def __init__(self, model_path, provenance_path):
           self.tracker = Tracker()
           self.verifier = SafetyVerifier()
           self.provenance_data = self.tracker.load_provenance(provenance_path)
           
       def monitor_safety(self, input_text):
           # Get current safety metrics
           current_metrics = self.tracker.get_current_safety_metrics()
           
           # Compare with training metrics
           safety_status = self.verifier.compare_with_training(
               current_metrics,
               self.provenance_data["safety_metrics"]
           )
           
           # Alert if safety standards are not met
           if not safety_status["meets_standards"]:
               self.alert_safety_violation(safety_status)
           
           return safety_status
   ```

#### Safety Requirements Verification

1. **Pre-deployment Verification**
   ```python
   def verify_deployment_requirements(model_path, provenance_path):
       # Load model and provenance
       model = load_model(model_path)
       provenance = load_provenance(provenance_path)
       
       # Verify safety metrics
       safety_metrics = provenance["safety_metrics"]
       if safety_metrics["passed_checks"] / safety_metrics["total_checks"] < 0.95:
           raise SafetyVerificationError("Model does not meet minimum safety requirements")
       
       # Verify content warning rate
       if safety_metrics["content_warnings"] > 100:
           raise SafetyVerificationError("Content warning rate exceeds acceptable threshold")
       
       return True
   ```

2. **Runtime Safety Monitoring**
   ```python
   class RuntimeSafetyMonitor:
       def __init__(self, model, safety_config):
           self.model = model
           self.safety_config = safety_config
           self.metrics = SafetyMetrics()
           
       def check_input(self, input_text):
           # Check input against safety requirements
           safety_result = self.model.safety_checker.check_input(input_text)
           
           # Update metrics
           self.metrics.update(safety_result)
           
           # Verify against training metrics
           if not self.verify_against_training():
               raise RuntimeSafetyError("Safety metrics deviate from training standards")
           
           return safety_result
   ```

3. **Safety Compliance Reporting**
   ```python
   def generate_compliance_report(model_path, provenance_path):
       # Load model and provenance
       model = load_model(model_path)
       provenance = load_provenance(provenance_path)
       
       # Generate compliance report
       report = {
           "model_id": model.model_id,
           "safety_metrics": {
               "pass_rate": f"{provenance['safety_metrics']['passed_checks'] / provenance['safety_metrics']['total_checks']:.2%}",
               "content_warnings": provenance["safety_metrics"]["content_warnings"],
               "failed_checks": provenance["safety_metrics"]["failed_checks"]
           },
           "compliance_status": {
               "meets_requirements": True,
               "verified_by": "Merkle Tree Validation",
               "verification_timestamp": datetime.now().isoformat()
           }
       }
       
       return report
   ```

#### Best Practices for Safety Verification

1. **Regular Verification**
   - Verify safety metrics before each deployment
   - Monitor safety metrics during runtime
   - Schedule periodic full safety audits

2. **Compliance Documentation**
   - Maintain detailed safety verification reports
   - Document any safety incidents or violations
   - Keep track of safety metric trends

3. **Automated Monitoring**
   - Implement continuous safety monitoring
   - Set up alerts for safety metric deviations
   - Automate safety verification in CI/CD pipeline

4. **User Safety Guidelines**
   - Provide clear safety requirements documentation
   - Include safety verification instructions
   - Document safety incident response procedures

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

#### Safety Verification Scenarios

1. **Pre-deployment Verification**
   ```python
   # Example: Verifying model before deployment
   def verify_model_deployment(model_path, provenance_path):
       # Load model and provenance
       model = load_model(model_path)
       provenance = load_provenance(provenance_path)
       
       # Verify safety metrics
       safety_metrics = provenance["safety_metrics"]
       
       # Check pass rate
       pass_rate = safety_metrics["passed_checks"] / safety_metrics["total_checks"]
       if pass_rate < 0.95:
           raise SafetyVerificationError(f"Pass rate {pass_rate:.2%} below threshold 95%")
       
       # Check content warnings
       if safety_metrics["content_warnings"] > 100:
           raise SafetyVerificationError(f"Content warnings {safety_metrics['content_warnings']} exceed limit 100")
       
       # Verify specific safety checks
       required_checks = ["age_rating", "content_filter", "sensitive_topics"]
       for check in required_checks:
           if check not in safety_metrics["passed_checks"]:
               raise SafetyVerificationError(f"Missing required safety check: {check}")
       
       return True
   ```

2. **Runtime Safety Monitoring**
   ```python
   # Example: Monitoring safety during model inference
   class RuntimeSafetyMonitor:
       def __init__(self, model, safety_config):
           self.model = model
           self.safety_config = safety_config
           self.metrics = SafetyMetrics()
           self.alert_threshold = 0.90  # 90% pass rate threshold
           
       def check_input(self, input_text):
           # Check input against safety requirements
           safety_result = self.model.safety_checker.check_input(input_text)
           
           # Update metrics
           self.metrics.update(safety_result)
           
           # Check against thresholds
           current_pass_rate = self.metrics.get_pass_rate()
           if current_pass_rate < self.alert_threshold:
               self.alert_safety_violation({
                   "type": "pass_rate_below_threshold",
                   "current_rate": current_pass_rate,
                   "threshold": self.alert_threshold
               })
           
           return safety_result
   ```

3. **Safety Incident Response**
   ```python
   # Example: Handling safety incidents
   class SafetyIncidentHandler:
       def __init__(self, model_path, provenance_path):
           self.model = load_model(model_path)
           self.provenance = load_provenance(provenance_path)
           self.incident_log = []
           
       def handle_safety_incident(self, incident):
           # Log incident
           self.incident_log.append({
               "timestamp": datetime.now().isoformat(),
               "incident": incident,
               "model_state": self.get_model_state()
           })
           
           # Check if incident requires model rollback
           if self.should_rollback(incident):
               self.rollback_model()
               
           # Generate incident report
           report = self.generate_incident_report(incident)
           
           # Notify stakeholders
           self.notify_stakeholders(report)
           
           return report
   ```

#### Safety Verification Diagrams

1. **Safety Verification Flow**
   ```
   [Model Training] → [Safety Metrics Collection] → [Merkle Tree Generation]
          ↓
   [Pre-deployment Verification] → [Deployment Decision]
          ↓
   [Runtime Monitoring] → [Safety Metrics Tracking]
          ↓
   [Incident Detection] → [Response & Mitigation]
   ```

2. **Safety Metrics Structure**
   ```
   Safety Metrics
   ├── Content Warnings
   │   ├── Age Rating
   │   ├── Content Filter
   │   └── Sensitive Topics
   ├── Passed Checks
   │   ├── Required Checks
   │   └── Optional Checks
   ├── Failed Checks
   │   ├── Critical Failures
   │   └── Non-critical Failures
   └── Total Checks
       ├── Training Checks
       └── Runtime Checks
   ```

#### CI/CD Integration

1. **GitHub Actions Workflow**
   ```yaml
   # .github/workflows/safety-verification.yml
   name: Model Safety Verification
   
   on:
     push:
       branches: [ main ]
     pull_request:
       branches: [ main ]
   
   jobs:
     safety-verification:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         
         - name: Set up Python
           uses: actions/setup-python@v2
           with:
             python-version: '3.9'
             
         - name: Install dependencies
           run: |
             python -m pip install --upgrade pip
             pip install -r requirements.txt
             
         - name: Run safety verification
           run: |
             python scripts/verify_safety.py \
               --model-path artifacts/models/latest \
               --provenance-path artifacts/provenance/latest.json \
               --min-pass-rate 0.95 \
               --max-content-warnings 100
   ```

2. **Jenkins Pipeline**
   ```groovy
   // Jenkinsfile
   pipeline {
       agent any
       
       stages {
           stage('Safety Verification') {
               steps {
                   sh '''
                       python scripts/verify_safety.py \
                           --model-path ${MODEL_PATH} \
                           --provenance-path ${PROVENANCE_PATH} \
                           --min-pass-rate 0.95 \
                           --max-content-warnings 100
                   '''
               }
           }
           
           stage('Deployment') {
               when {
                   expression { 
                       return currentBuild.result == 'SUCCESS' 
                   }
               }
               steps {
                   sh '''
                       python scripts/deploy_model.py \
                           --model-path ${MODEL_PATH} \
                           --provenance-path ${PROVENANCE_PATH}
                   '''
               }
           }
       }
       
       post {
           failure {
               emailext (
                   subject: "Safety Verification Failed",
                   body: "Safety verification failed for model ${MODEL_PATH}",
                   to: 'team@example.com'
               )
           }
       }
   }
   ```

3. **Automated Safety Testing**
   ```python
   # tests/test_safety_verification.py
   import pytest
   from ml_provenance.safety.verifier import SafetyVerifier
   
   @pytest.fixture
   def safety_verifier():
       return SafetyVerifier()
   
   def test_safety_metrics_verification(safety_verifier):
       # Test case 1: Valid safety metrics
       valid_metrics = {
           "content_warnings": 50,
           "passed_checks": 950,
           "total_checks": 1000
       }
       assert safety_verifier.verify_metrics(valid_metrics) == True
       
       # Test case 2: Invalid safety metrics
       invalid_metrics = {
           "content_warnings": 150,
           "passed_checks": 800,
           "total_checks": 1000
       }
       assert safety_verifier.verify_metrics(invalid_metrics) == False
   
   def test_runtime_safety_monitoring(safety_verifier):
       # Test case: Runtime safety monitoring
       input_text = "Test input"
       safety_result = safety_verifier.check_input(input_text)
       
       assert "content_warnings" in safety_result
       assert "passed_checks" in safety_result
       assert safety_result["passed_checks"] >= 0
   ```

#### Safety Verification Best Practices

1. **Automated Testing**
   - Implement unit tests for safety verification
   - Add integration tests for safety monitoring
   - Include safety tests in CI/CD pipeline

2. **Monitoring and Alerting**
   - Set up real-time safety monitoring
   - Configure alerts for safety violations
   - Implement automated incident response

3. **Documentation and Reporting**
   - Maintain detailed safety verification logs
   - Generate regular safety reports
   - Document safety incidents and responses

4. **Continuous Improvement**
   - Regularly review safety thresholds
   - Update safety checks based on new requirements
   - Incorporate feedback from safety incidents

## Additional Safety Features

1. **Safety Verification Scenarios**
   - Pre-deployment verification
   - Runtime monitoring
   - Safety incident response

2. **Safety Verification Diagrams**
   - Safety verification flow
   - Safety metrics structure

3. **CI/CD Integration**
   - GitHub Actions workflow
   - Jenkins pipeline
   - Automated safety testing

4. **Safety Verification Best Practices**
   - Automated testing
   - Monitoring and alerting
   - Documentation and reporting
   - Continuous improvement

#### Safety Incident Scenarios and Handling

1. **Content Warning Threshold Exceeded**
   ```python
   # Example: Handling excessive content warnings
   class ContentWarningHandler:
       def __init__(self, threshold=100):
           self.threshold = threshold
           self.warning_log = []
           
       def handle_content_warning(self, warning):
           self.warning_log.append({
               "timestamp": datetime.now().isoformat(),
               "warning": warning,
               "severity": self.calculate_severity(warning)
           })
           
           if len(self.warning_log) > self.threshold:
               return {
                   "action": "block_generation",
                   "reason": "Content warning threshold exceeded",
                   "warnings": self.warning_log[-10:],  # Last 10 warnings
                   "total_warnings": len(self.warning_log)
               }
           return {"action": "continue", "warning_count": len(self.warning_log)}
   ```

2. **Safety Check Failure**
   ```python
   # Example: Handling safety check failures
   class SafetyCheckHandler:
       def __init__(self, model, provenance):
           self.model = model
           self.provenance = provenance
           self.failure_log = []
           
       def handle_safety_failure(self, failure):
           # Log failure
           self.failure_log.append({
               "timestamp": datetime.now().isoformat(),
               "failure_type": failure["type"],
               "details": failure["details"],
               "model_state": self.get_model_state()
           })
           
           # Check if failure is critical
           if self.is_critical_failure(failure):
               return {
                   "action": "rollback",
                   "reason": "Critical safety check failure",
                   "failure_details": failure,
                   "recommended_action": "Model rollback required"
               }
           
           # For non-critical failures
           return {
               "action": "warn",
               "reason": "Non-critical safety check failure",
               "failure_details": failure,
               "recommended_action": "Monitor and log"
           }
   ```

3. **Model Drift Detection**
   ```python
   # Example: Handling model drift
   class ModelDriftHandler:
       def __init__(self, baseline_metrics, drift_threshold=0.1):
           self.baseline_metrics = baseline_metrics
           self.drift_threshold = drift_threshold
           self.drift_log = []
           
       def detect_drift(self, current_metrics):
           drift_scores = {}
           for metric in self.baseline_metrics:
               drift = abs(current_metrics[metric] - self.baseline_metrics[metric])
               drift_scores[metric] = drift
               
               if drift > self.drift_threshold:
                   self.drift_log.append({
                       "timestamp": datetime.now().isoformat(),
                       "metric": metric,
                       "drift": drift,
                       "baseline": self.baseline_metrics[metric],
                       "current": current_metrics[metric]
                   })
           
           return {
               "has_drift": any(drift > self.drift_threshold for drift in drift_scores.values()),
               "drift_scores": drift_scores,
               "drift_log": self.drift_log
           }
   ```

#### Detailed Safety Verification Diagrams

1. **Safety Incident Response Flow**
   ```
   [Incident Detection]
          ↓
   [Severity Assessment]
          ↓
   [Critical?] → No → [Log & Monitor]
          ↓ Yes
   [Immediate Actions]
          ↓
   [Model State Check]
          ↓
   [Rollback Required?] → No → [Mitigation Actions]
          ↓ Yes
   [Model Rollback]
          ↓
   [Incident Report]
          ↓
   [Stakeholder Notification]
   ```

2. **Safety Metrics Monitoring**
   ```
   [Real-time Metrics]
          ↓
   [Threshold Check]
          ↓
   [Below Threshold?] → Yes → [Normal Operation]
          ↓ No
   [Alert Generation]
          ↓
   [Incident Handler]
          ↓
   [Response Actions]
          ↓
   [Metrics Update]
   ```

3. **Model Validation Process**
   ```
   [Model Training]
          ↓
   [Safety Metrics Collection]
          ↓
   [Merkle Tree Generation]
          ↓
   [Pre-deployment Checks]
          ↓
   [Safety Verification]
          ↓
   [Deployment Decision]
          ↓
   [Runtime Monitoring]
   ```

#### Additional Test Cases

1. **Safety Metrics Validation**
   ```python
   # tests/test_safety_metrics.py
   import pytest
   from ml_provenance.safety.metrics import SafetyMetrics
   
   @pytest.fixture
   def safety_metrics():
       return SafetyMetrics()
   
   def test_content_warning_threshold(safety_metrics):
       # Test case: Content warning threshold
       for _ in range(95):
           safety_metrics.add_warning("test_warning")
       
       assert safety_metrics.get_warning_count() == 95
       assert not safety_metrics.is_threshold_exceeded()
       
       # Add more warnings to exceed threshold
       for _ in range(10):
           safety_metrics.add_warning("test_warning")
       
       assert safety_metrics.is_threshold_exceeded()
   
   def test_safety_check_failure(safety_metrics):
       # Test case: Safety check failure
       safety_metrics.record_check("age_rating", False)
       safety_metrics.record_check("content_filter", True)
       
       assert safety_metrics.get_failure_count() == 1
       assert safety_metrics.get_pass_rate() == 0.5
   
   def test_model_drift(safety_metrics):
       # Test case: Model drift detection
       baseline = {
           "pass_rate": 0.95,
           "warning_rate": 0.05
       }
       
       current = {
           "pass_rate": 0.85,
           "warning_rate": 0.15
       }
       
       drift = safety_metrics.calculate_drift(baseline, current)
       assert drift["pass_rate"] > 0.1  # Significant drift
       assert drift["warning_rate"] > 0.1  # Significant drift
   ```

2. **Incident Response Testing**
   ```python
   # tests/test_incident_response.py
   import pytest
   from ml_provenance.safety.incident import IncidentHandler
   
   @pytest.fixture
   def incident_handler():
       return IncidentHandler()
   
   def test_critical_incident(incident_handler):
       # Test case: Critical incident handling
       incident = {
           "type": "critical_safety_failure",
           "details": {
               "check": "age_rating",
               "severity": "high",
               "impact": "model_rollback_required"
           }
       }
       
       response = incident_handler.handle_incident(incident)
       assert response["action"] == "rollback"
       assert response["severity"] == "high"
       assert "rollback_required" in response["recommended_action"]
   
   def test_warning_threshold(incident_handler):
       # Test case: Warning threshold handling
       for _ in range(95):
           incident_handler.handle_warning("test_warning")
       
       assert not incident_handler.is_threshold_exceeded()
       
       # Add more warnings
       for _ in range(10):
           incident_handler.handle_warning("test_warning")
       
       assert incident_handler.is_threshold_exceeded()
       assert incident_handler.get_action() == "block_generation"
   ```

3. **Integration Testing**
   ```python
   # tests/test_integration.py
   import pytest
   from ml_provenance.safety.verifier import SafetyVerifier
   from ml_provenance.safety.monitor import SafetyMonitor
   
   @pytest.fixture
   def safety_system():
       verifier = SafetyVerifier()
       monitor = SafetyMonitor()
       return {"verifier": verifier, "monitor": monitor}
   
   def test_end_to_end_safety(safety_system):
       # Test case: End-to-end safety verification
       model_path = "artifacts/models/latest"
       provenance_path = "artifacts/provenance/latest.json"
       
       # Verify model
       verification_result = safety_system["verifier"].verify_model(
           model_path,
           provenance_path
       )
       assert verification_result["status"] == "passed"
       
       # Monitor runtime
       input_text = "Test input"
       safety_result = safety_system["monitor"].check_input(input_text)
       assert safety_result["status"] == "safe"
       assert safety_result["warnings"] == []
   ``` 