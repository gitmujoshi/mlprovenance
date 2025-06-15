# User Documentation

## Overview

This document provides instructions for using the ML provenance tracking and safety features system.

[Source: `safety_features/app/app.py`]

## 1. Getting Started

### 1.1 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist_provenance.git
   cd mnist_provenance
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

[Source: `requirements.txt`]

### 1.2 Configuration

1. Set up the provenance directory:
   ```bash
   mkdir -p artifacts/provenance
   ```

2. Configure safety features:
   ```bash
   cp safety_features/app/config.example.py safety_features/app/config.py
   ```

3. Edit the configuration file to set your preferences:
   ```python
   # safety_features/app/config.py
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

[Source: `safety_features/app/config.py`]

## 2. Using the Web Interface

### 2.1 Starting the Application

1. Start the Flask application:
   ```bash
   python safety_features/scripts/run_app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5001
   ```

[Source: `safety_features/scripts/run_app.py`]

### 2.2 Generating Text

1. Enter your prompt in the text input field
2. Click "Generate" to create text
3. The system will:
   - Check the input for safety violations
   - Generate text using the model
   - Verify the output against safety rules
   - Display the result with safety metrics

[Source: `safety_features/app/templates/index.html`]

### 2.3 Viewing Safety Metrics

1. Click on "Safety Metrics" in the navigation bar
2. View:
   - Content warning statistics
   - Safety check pass rates
   - Recent violations
   - Model safety status

[Source: `safety_features/app/templates/safety_metrics.html`]

## 3. Command Line Interface

### 3.1 Training with Safety Features

```bash
python safety_features/scripts/train_gpt2_with_safety.py \
    --model_name gpt2 \
    --dataset wikitext-2-raw-v1 \
    --epochs 3 \
    --batch_size 4 \
    --safety_config safety_features/app/config.py
```

[Source: `safety_features/scripts/train_gpt2_with_safety.py`]

### 3.2 Verifying Model Provenance

```bash
python scripts/generate_merkle_hash_report.py \
    --model_path artifacts/provenance/run_20250615_153833/trained_model.pt \
    --provenance_path artifacts/provenance/run_20250615_153833/provenance_report_20250615_153901.json
```

[Source: `scripts/generate_merkle_hash_report.py`]

## 4. Safety Features

### 4.1 Content Filtering

The system implements several content filters:

1. **Profanity Filter**
   - Checks for inappropriate language
   - Configurable word lists
   - Context-aware filtering

2. **Sensitive Topics**
   - Identifies potentially sensitive content
   - Age-appropriate filtering
   - Customizable topic lists

3. **Age Rating**
   - Assigns age ratings to content
   - Enforces age restrictions
   - Configurable rating levels

[Source: `safety_features/app/safety/filters.py`]

### 4.2 Safety Metrics

The system tracks various safety metrics:

1. **Content Warnings**
   - Number of warnings generated
   - Types of violations
   - Warning severity levels

2. **Safety Check Results**
   - Pass/fail rates
   - Check types
   - Historical trends

3. **Model Safety Status**
   - Overall safety score
   - Compliance status
   - Safety thresholds

[Source: `safety_features/app/safety/metrics.py`]

## 5. Provenance Tracking

### 5.1 Viewing Provenance Data

1. Navigate to the provenance directory:
   ```bash
   cd artifacts/provenance
   ```

2. View the latest run:
   ```bash
   ls -l run_*/provenance_report_*.json
   ```

3. Examine the report:
   ```bash
   cat run_*/provenance_report_*.json | jq
   ```

[Source: `artifacts/provenance/run_20250615_153833/provenance_report_20250615_153901.json`]

### 5.2 Verifying Model Integrity

1. Generate a verification report:
   ```bash
   python scripts/generate_training_report.py
   ```

2. Check the report at:
   ```
   docs/training_run_report.md
   ```

[Source: `scripts/generate_training_report.py`]

## 6. Troubleshooting

### 6.1 Common Issues

1. **Port Already in Use**
   - Error: "Address already in use"
   - Solution: Change the port in `safety_features/scripts/run_app.py`
   - Alternative: Disable AirPlay Receiver on macOS

2. **Model Loading Errors**
   - Error: "Failed to load model"
   - Solution: Check model path and file permissions
   - Verify model file integrity

3. **Safety Check Failures**
   - Error: "Safety check failed"
   - Solution: Review safety configuration
   - Check input content

[Source: `safety_features/app/errors.py`]

### 6.2 Getting Help

1. Check the logs:
   ```bash
   tail -f safety_features/app/logs/app.log
   ```

2. Review error messages in the web interface

3. Contact support with:
   - Error messages
   - Log files
   - System information

[Source: `safety_features/app/logger.py`]

## 7. Best Practices

### 7.1 Safety Guidelines

1. **Content Generation**
   - Review generated content
   - Monitor safety metrics
   - Report violations

2. **Model Usage**
   - Verify model provenance
   - Check safety status
   - Update regularly

3. **Configuration**
   - Regular safety updates
   - Monitor thresholds
   - Backup settings

[Source: `safety_features/app/config.py`]

### 7.2 Maintenance

1. **Regular Updates**
   - Update dependencies
   - Check for new safety rules
   - Verify model integrity

2. **Backup**
   - Backup configuration
   - Save provenance data
   - Archive safety reports

3. **Monitoring**
   - Check safety metrics
   - Review error logs
   - Update documentation

[Source: `safety_features/scripts/maintenance.py`]

## 1. User Guides

### 1.1 Getting Started

#### 1.1.1 Installation
```bash
# Install the ML Provenance package
pip install ml-provenance

# Initialize the system
ml-provenance init
```

#### 1.1.2 Basic Usage
```python
# Basic tracking example
from ml_provenance import ProvenanceTracker

# Initialize tracker
tracker = ProvenanceTracker()

# Track data
with tracker.track_data():
    data = load_dataset()
    tracker.log_data(data)

# Track model
with tracker.track_model():
    model = train_model()
    tracker.log_model(model)
```

### 1.2 Advanced Usage

#### 1.2.1 Custom Tracking
```python
# Custom tracking example
from ml_provenance import ProvenanceTracker

tracker = ProvenanceTracker()

# Custom metadata
metadata = {
    'project': 'my_project',
    'version': '1.0.0',
    'description': 'Custom tracking example'
}

# Track with custom metadata
with tracker.track_experiment(metadata=metadata):
    # Your ML workflow here
    pass
```

## 2. Best Practices

### 2.1 Data Management
- Use consistent naming conventions
- Document data preprocessing steps
- Maintain data versioning
- Regular data validation

### 2.2 Model Management
- Version all model changes
- Document model architecture
- Track hyperparameters
- Monitor model performance

### 2.3 Experiment Management
- Use descriptive experiment names
- Document experiment configurations
- Track all dependencies
- Regular experiment cleanup

### 2.4 Security Practices
- Secure API keys
- Regular access review
- Data encryption
- Audit logging

## 3. Troubleshooting Guides

### 3.1 Common Issues

#### 3.1.1 Installation Issues
- **Problem**: Package installation fails
- **Solution**: 
  1. Check Python version compatibility
  2. Verify pip installation
  3. Check system dependencies

#### 3.1.2 Connection Issues
- **Problem**: Cannot connect to tracking server
- **Solution**:
  1. Verify network connectivity
  2. Check server status
  3. Validate credentials

#### 3.1.3 Performance Issues
- **Problem**: Slow tracking operations
- **Solution**:
  1. Check system resources
  2. Optimize batch operations
  3. Review storage configuration

### 3.2 Error Messages

#### 3.2.1 Common Error Codes
- E001: Authentication failed
- E002: Invalid data format
- E003: Storage quota exceeded
- E004: Version conflict

#### 3.2.2 Resolution Steps
1. Check error message details
2. Review system logs
3. Verify configuration
4. Contact support if needed

### 3.3 Performance Optimization

#### 3.3.1 System Tuning
- Optimize batch sizes
- Configure caching
- Adjust storage settings
- Monitor resource usage

#### 3.3.2 Best Practices
- Regular system maintenance
- Performance monitoring
- Resource optimization
- Regular updates 

## When Should You Run the Verifier?

The verifier is a tool that checks the integrity and authenticity of your data, model, and training process. Here are the main situations when you should use it:

1. **Before Deploying a Model**
   - Make sure your model and data haven't been tampered with before going live.

2. **During Audits or Compliance Checks**
   - Prove to auditors or regulators that your model's history and data are trustworthy.

3. **After Training a Model**
   - Confirm that your training process was correct and can be reproduced.

4. **When Sharing or Transferring Models**
   - Let others verify that the model and its history are authentic and unchanged.

5. **Before or After Model Updates/Retraining**
   - Ensure that updates or retraining haven't broken the chain of trust.

6. **When Investigating Issues or Anomalies**
   - Check for unauthorized changes or data corruption if something goes wrong.

**Summary Table:**

| When to Run Verifier         | Why/Goal                                      |
|-----------------------------|------------------------------------------------|
| Before deployment           | Ensure integrity before production use         |
| During audits/compliance    | Satisfy regulatory or internal requirements    |
| After training              | Confirm reproducibility and correctness        |
| When sharing/transferring   | Build trust and transparency                   |
| Before/after updates        | Maintain chain of trust across versions        |
| During incident investigation| Detect tampering or corruption                |

**In short:**
Run the verifier whenever you need to check or prove the trustworthiness of your ML pipeline, especially before deployment, during audits, or when sharing models. 