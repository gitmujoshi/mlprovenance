# Technical Documentation

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