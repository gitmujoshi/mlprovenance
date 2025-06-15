# Technical Documentation

## Overview

This document provides technical details about the implementation of the ML provenance tracking and safety features system.

[Source: `src/ml_provenance/provenance/tracker.py`]

## 1. System Architecture

### 1.1 Core Components

1. **Provenance Tracker**
   - Tracks data, model, and training provenance
   - Generates and verifies Merkle trees
   - Manages provenance data storage

[Source: `src/ml_provenance/provenance/tracker.py` - `ProvenanceTracker` class]

2. **Safety Features**
   - Implements content filtering
   - Manages age ratings
   - Tracks safety metrics

[Source: `safety_features/app/app.py` - Safety implementation]

3. **Verification System**
   - Validates model integrity
   - Verifies training provenance
   - Ensures safety compliance

[Source: `src/ml_provenance/provenance/verifier.py` - `ProvenanceVerifier` class]

### 1.2 Data Flow

```
[Data Source] → [Preprocessing] → [Training] → [Model]
     ↓              ↓               ↓           ↓
[Provenance] → [Merkle Tree] → [Verification] → [Deployment]
     ↓              ↓               ↓           ↓
[Safety Checks] → [Metrics] → [Reports] → [Monitoring]
```

[Source: `safety_features/scripts/train_gpt2_with_safety.py` - Training pipeline]

## 2. Implementation Details

### 2.1 Provenance Tracking

The provenance tracking system is implemented using a Merkle tree structure to ensure data integrity and traceability.

[Source: `src/ml_provenance/provenance/merkle_tree.py` - `MLProvenanceMerkleTree` class]

```python
class MLProvenanceMerkleTree:
    def __init__(self):
        self.nodes = {}
        self.root = None
        
    def add_node(self, component: str, data: Dict[str, Any]) -> None:
        # Generate hash for the component
        component_hash = self._hash_component(component, data)
        
        # Create node
        node = {
            "component": component,
            "hash": component_hash,
            "timestamp": data.get("timestamp", ""),
            "children": []
        }
        
        # Add to nodes
        self.nodes[component] = node
```

### 2.2 Safety Features

The safety features are implemented as a set of checks and filters that are applied during model training and inference.

[Source: `safety_features/app/app.py` - Safety implementation]

```python
class SafetyChecker:
    def __init__(self, config):
        self.config = config
        self.filters = self._load_filters()
        
    def check_input(self, text: str) -> Dict[str, Any]:
        results = {
            "passed": True,
            "warnings": [],
            "violations": []
        }
        
        # Apply content filters
        for filter_name, filter_func in self.filters.items():
            if not filter_func(text):
                results["passed"] = False
                results["violations"].append(filter_name)
                
        return results
```

### 2.3 Verification System

The verification system ensures that all components of the ML pipeline maintain their integrity and can be traced back to their origin.

[Source: `src/ml_provenance/provenance/verifier.py` - `ProvenanceVerifier` class]

```python
class ProvenanceVerifier:
    def __init__(self, provenance_dir: Optional[Path] = None):
        self.provenance_data = {
            "data": {},
            "model": {},
            "training": {}
        }
        self.provenance_dir = Path(provenance_dir) if provenance_dir else None
        
    def verify_model(self, model_path: Path) -> Dict[str, Any]:
        results = {
            "model_exists": True,
            "model_hash_match": False,
            "model_merkle_verified": False,
            "architecture_verified": False
        }
        
        # Verify model hash
        if "hash" in self.provenance_data["model"]:
            current_hash = self._generate_hash(model_path)
            results["model_hash_match"] = current_hash == self.provenance_data["model"]["hash"]
            
        return results
```

## 3. API Reference

### 3.1 Provenance Tracker API

[Source: `src/ml_provenance/provenance/tracker.py` - `ProvenanceTracker` class]

```python
class ProvenanceTracker:
    def track_data(self, data_path: Path) -> Dict[str, Any]:
        """Track data provenance."""
        pass
        
    def track_model(self, model_path: Path) -> Dict[str, Any]:
        """Track model provenance."""
        pass
        
    def track_training(self, training_path: Path) -> Dict[str, Any]:
        """Track training provenance."""
        pass
```

### 3.2 Safety Features API

[Source: `safety_features/app/app.py` - Safety implementation]

```python
class SafetyFeatures:
    def check_content(self, text: str) -> Dict[str, Any]:
        """Check content for safety violations."""
        pass
        
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get current safety metrics."""
        pass
        
    def update_safety_config(self, config: Dict[str, Any]) -> None:
        """Update safety configuration."""
        pass
```

### 3.3 Verification API

[Source: `src/ml_provenance/provenance/verifier.py` - `ProvenanceVerifier` class]

```python
class ProvenanceVerifier:
    def verify_data(self, data_path: Path) -> Dict[str, Any]:
        """Verify data provenance."""
        pass
        
    def verify_model(self, model_path: Path) -> Dict[str, Any]:
        """Verify model provenance."""
        pass
        
    def verify_training(self, training_path: Path) -> Dict[str, Any]:
        """Verify training provenance."""
        pass
```

## 4. Configuration

### 4.1 Provenance Configuration

[Source: `src/ml_provenance/provenance/config.py`]

```python
PROVENANCE_CONFIG = {
    "hash_function": "sha256",
    "storage_path": "artifacts/provenance",
    "verification_interval": 3600,  # seconds
    "max_history": 1000
}
```

### 4.2 Safety Configuration

[Source: `safety_features/app/config.py`]

```python
SAFETY_CONFIG = {
    "content_filters": [
        "profanity",
        "sensitive_topics",
        "age_rating"
    ],
    "min_pass_rate": 0.95,
    "max_content_warnings": 100
}
```

## 5. Error Handling

### 5.1 Provenance Errors

[Source: `src/ml_provenance/provenance/errors.py`]

```python
class ProvenanceError(Exception):
    """Base class for provenance errors."""
    pass

class HashMismatchError(ProvenanceError):
    """Raised when hash verification fails."""
    pass

class MissingComponentError(ProvenanceError):
    """Raised when a required component is missing."""
    pass
```

### 5.2 Safety Errors

[Source: `safety_features/app/errors.py`]

```python
class SafetyError(Exception):
    """Base class for safety errors."""
    pass

class ContentViolationError(SafetyError):
    """Raised when content violates safety rules."""
    pass

class SafetyConfigError(SafetyError):
    """Raised when safety configuration is invalid."""
    pass
```

## 6. Logging

### 6.1 Provenance Logging

[Source: `src/ml_provenance/provenance/logger.py`]

```python
class ProvenanceLogger:
    def __init__(self):
        self.logger = logging.getLogger('provenance')
        
    def log_tracking(self, component: str, data: Dict[str, Any]):
        self.logger.info(f"Tracking {component}")
        self.logger.debug(f"Data: {data}")
        
    def log_verification(self, component: str, result: Dict[str, Any]):
        self.logger.info(f"Verifying {component}")
        self.logger.debug(f"Result: {result}")
```

### 6.2 Safety Logging

[Source: `safety_features/app/logger.py`]

```python
class SafetyLogger:
    def __init__(self):
        self.logger = logging.getLogger('safety')
        
    def log_check(self, text: str, result: Dict[str, Any]):
        self.logger.info(f"Safety check for text: {text[:100]}...")
        self.logger.debug(f"Result: {result}")
        
    def log_violation(self, text: str, violation: str):
        self.logger.warning(f"Safety violation: {violation}")
        self.logger.debug(f"Text: {text[:100]}...")
```

## 7. Testing

### 7.1 Provenance Tests

[Source: `tests/test_provenance.py`]

```python
def test_provenance_tracking():
    tracker = ProvenanceTracker()
    data = {"test": "data"}
    result = tracker.track_data(data)
    assert result["hash"] is not None
    assert result["timestamp"] is not None
```

### 7.2 Safety Tests

[Source: `tests/test_safety.py`]

```python
def test_safety_checks():
    checker = SafetyChecker()
    result = checker.check_input("test text")
    assert result["passed"] is True
    assert len(result["warnings"]) == 0
```

## 8. Deployment

### 8.1 Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.0+
- Flask 2.0+

### 8.2 Installation

```bash
pip install -r requirements.txt
```

### 8.3 Configuration

1. Set up provenance directory:
   ```bash
   mkdir -p artifacts/provenance
   ```

2. Configure safety features:
   ```bash
   cp safety_features/app/config.example.py safety_features/app/config.py
   ```

3. Start the application:
   ```bash
   python safety_features/scripts/run_app.py
   ```

[Source: `safety_features/scripts/run_app.py`]

## Package Structure

The project is organized as a Python package with the following structure:

```
ml_provenance/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── mnist_data.py
├── models/
│   ├── __init__.py
│   └── mnist_model.py
├── provenance/
│   ├── __init__.py
│   ├── tracker.py
│   ├── verifier.py
│   ├── merkle_tree.py
│   └── generate_final_report.py
└── training/
    ├── __init__.py
    └── train.py
```

## Report Generation

The report generation system provides a comprehensive view of the model's provenance and training process. Reports are generated in JSON format and include:

- Timestamp of report generation
- Model provenance verification results
- Data provenance verification results
- Training configuration
- Overall verification status

### Usage

```python
from ml_provenance.provenance.generate_final_report import generate_final_report

report_path = generate_final_report(
    model_path="path/to/model.pt",
    data_path="path/to/data.pt",
    training_config={
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "privacy_epsilon": 1.0,
        "privacy_delta": 1e-5
    },
    output_dir="artifacts/provenance"
)
```

### Report Structure

The generated report has the following structure:

```json
{
    "timestamp": "2024-03-14T12:00:00",
    "model_provenance": {
        "verified": true,
        "hash": "...",
        "details": {
            "architecture": "...",
            "parameters": 123456
        }
    },
    "data_provenance": {
        "verified": true,
        "hash": "...",
        "details": {
            "dataset": "MNIST",
            "samples": 60000
        }
    },
    "training_config": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "privacy_epsilon": 1.0,
        "privacy_delta": 1e-5
    },
    "verification_status": "PASSED"
}
```

### Testing

The report generation system includes comprehensive tests in `tests/test_report_generation.py`. Tests cover:

- Directory and file creation
- Report structure validation
- Timestamp format verification
- Configuration matching

To run the tests:

```bash
pytest tests/test_report_generation.py -v
```

## 1. System Architecture

### 1.1 Overview
The ML Provenance System is built on a three-layer architecture:
- Data Layer
- Model Layer
- Experiment Layer

### 1.2 Component Architecture

#### 1.2.1 Data Layer
- **Data Ingestion Service**
  - Handles raw data input
  - Validates data format
  - Generates initial hashes

- **Data Processing Service**
  - Manages data transformations
  - Tracks preprocessing steps
  - Maintains data lineage

- **Data Storage Service**
  - Manages data persistence
  - Handles data versioning
  - Controls data access

#### 1.2.2 Model Layer
- **Model Registry**
  - Stores model artifacts
  - Manages model versions
  - Tracks model metadata

- **Model Training Service**
  - Handles model training
  - Tracks training metrics
  - Manages model checkpoints

- **Model Deployment Service**
  - Manages model deployment
  - Tracks deployment history
  - Handles model rollbacks

#### 1.2.3 Experiment Layer
- **Experiment Manager**
  - Tracks experiment configurations
  - Manages experiment runs
  - Stores experiment results

- **Metrics Collector**
  - Collects performance metrics
  - Tracks resource usage
  - Monitors system health

### 1.3 System Interactions
- Inter-service communication
- Data flow patterns
- Event handling
- Error management

## 2. API Documentation

### 2.1 REST API Endpoints

#### 2.1.1 Data Management
```
POST /api/v1/data
- Upload new dataset
- Parameters: file, metadata
- Returns: dataset_id

GET /api/v1/data/{dataset_id}
- Retrieve dataset information
- Returns: dataset metadata

PUT /api/v1/data/{dataset_id}
- Update dataset metadata
- Parameters: metadata
- Returns: updated dataset info
```

#### 2.1.2 Model Management
```
POST /api/v1/models
- Register new model
- Parameters: model_file, metadata
- Returns: model_id

GET /api/v1/models/{model_id}
- Get model information
- Returns: model metadata

PUT /api/v1/models/{model_id}/version
- Create new model version
- Parameters: version_info
- Returns: version_id
```

#### 2.1.3 Experiment Management
```
POST /api/v1/experiments
- Start new experiment
- Parameters: config
- Returns: experiment_id

GET /api/v1/experiments/{experiment_id}
- Get experiment results
- Returns: experiment data

PUT /api/v1/experiments/{experiment_id}/status
- Update experiment status
- Parameters: status
- Returns: updated status
```

### 2.2 Authentication and Authorization
- API key management
- Role-based access control
- Token-based authentication
- Permission levels

### 2.3 Error Handling
- Error codes
- Error messages
- Retry policies
- Rate limiting

## 3. Integration Guides

### 3.1 ML Framework Integration

#### 3.1.1 TensorFlow Integration
```python
# Example TensorFlow integration
from ml_provenance import ProvenanceTracker

tracker = ProvenanceTracker()
with tracker.track_experiment():
    model = tf.keras.Sequential([...])
    model.fit(...)
    tracker.log_model(model)
```

#### 3.1.2 PyTorch Integration
```python
# Example PyTorch integration
from ml_provenance import ProvenanceTracker

tracker = ProvenanceTracker()
with tracker.track_experiment():
    model = torch.nn.Sequential(...)
    trainer.fit(model)
    tracker.log_model(model)
```

### 3.2 Storage System Integration

#### 3.2.1 Database Integration
- Connection configuration
- Schema setup
- Query optimization
- Backup procedures

#### 3.2.2 File System Integration
- Directory structure
- File naming conventions
- Access patterns
- Cleanup procedures

### 3.3 Monitoring Integration
- Metrics collection
- Alert configuration
- Dashboard setup
- Log management 

## When to Run the Verifier

The verifier in an ML provenance system is used to check the integrity, authenticity, and reproducibility of the data, model, and training process. Here are the key scenarios when you should run the verifier:

### 1. Before Model Deployment
- **Purpose:** Ensure the model and its training data have not been tampered with and match the expected provenance.
- **Why:** Prevent deploying a model that may have been altered, corrupted, or trained on unapproved data.

### 2. During Audits or Compliance Checks
- **Purpose:** Demonstrate to auditors or regulators that the model's lineage and training process are intact and verifiable.
- **Why:** Many industries (finance, healthcare, etc.) require proof of data and model integrity for compliance.

### 3. After Model Training (for Reproducibility)
- **Purpose:** Confirm that the training process produced the expected results and that the model can be reproduced from the recorded provenance.
- **Why:** Ensures scientific rigor and supports claims of reproducibility.

### 4. When Sharing or Transferring Models
- **Purpose:** Allow recipients to verify that the model and its provenance are authentic and unchanged.
- **Why:** Builds trust and transparency when models are shared between teams, organizations, or published.

### 5. Before/After Model Updates or Retraining
- **Purpose:** Verify that updates or retraining have not introduced inconsistencies or errors in the provenance chain.
- **Why:** Maintains a continuous chain of trust and integrity across model versions.

### 6. When Investigating Anomalies or Incidents
- **Purpose:** Check if any unauthorized changes or data corruption have occurred that could explain unexpected model behavior.
- **Why:** Supports root-cause analysis and incident response.

#### Summary Table

| When to Run Verifier          | Why/Goal                                      |
|------------------------------|-----------------------------------------------|
| Before deployment            | Ensure integrity before production use         |
| During audits/compliance     | Satisfy regulatory or internal requirements    |
| After training               | Confirm reproducibility and correctness        |
| When sharing/transferring    | Build trust and transparency                   |
| Before/after updates         | Maintain chain of trust across versions        |
| During incident investigation| Detect tampering or corruption                 |

**In short:**
Run the verifier whenever you need to prove, check, or trust the integrity and authenticity of your ML pipeline, especially at critical handoff, deployment, or compliance points. 