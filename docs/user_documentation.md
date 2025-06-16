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

## 3. Training Models with Safety Features

### 3.1 Basic Training Setup

1. Create a training configuration file (e.g., `train_config.py`):
   ```python
   TRAINING_CONFIG = {
       "model": {
           "type": "transformer",  # or "cnn", "rnn", "custom"
           "architecture": "bert-base-uncased",  # model identifier or path
           "custom_config": {}  # optional custom model configuration
       },
       "training": {
           "epochs": 3,
           "batch_size": 32,
           "learning_rate": 1e-4,
           "optimizer": "adam",
           "scheduler": "cosine"
       },
       "safety": {
           "enabled": True,
           "config_path": "safety_features/app/config.py"
       }
   }
   ```

2. Run training with safety features:
   ```bash
   python safety_features/scripts/train_model_with_safety.py \
       --model_name <your_model_name> \
       --config_path train_config.py \
       --safety_config safety_features/app/config.py
   ```

### 3.2 Framework-Specific Training

#### 3.2.1 PyTorch Training

1. Create a PyTorch model with safety features:
   ```python
   import torch
   import torch.nn as nn
   from safety_features.frameworks.pytorch import PyTorchSafetyWrapper

   class SafePyTorchModel(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.model = YourPyTorchModel()
           self.safety_wrapper = PyTorchSafetyWrapper(config)
           
       def forward(self, x):
           # Safety checks before forward pass
           x = self.safety_wrapper.validate_input(x)
           
           # Regular forward pass
           output = self.model(x)
           
           # Safety checks after forward pass
           output = self.safety_wrapper.validate_output(output)
           return output
   ```

2. Configure PyTorch training:
   ```python
   pytorch_config = {
       "framework": "pytorch",
       "safety": {
           "enabled": True,
           "checks": {
               "input_validation": True,
               "output_validation": True,
               "gradient_clipping": True
           },
           "metrics": {
               "track_gradients": True,
               "track_activations": True
           }
       },
       "training": {
           "safety_weight": 0.1,
           "gradient_clip_val": 1.0
       }
   }
   ```

3. Run PyTorch training:
   ```bash
   python safety_features/scripts/train_model_with_safety.py \
       --model_name SafePyTorchModel \
       --config_path pytorch_config.py \
       --framework pytorch
   ```

#### 3.2.2 TensorFlow Training

1. Create a TensorFlow model with safety features:
   ```python
   import tensorflow as tf
   from safety_features.frameworks.tensorflow import TensorFlowSafetyWrapper

   class SafeTensorFlowModel(tf.keras.Model):
       def __init__(self, config):
           super().__init__()
           self.model = YourTensorFlowModel()
           self.safety_wrapper = TensorFlowSafetyWrapper(config)
           
       def call(self, inputs, training=False):
           # Safety checks before forward pass
           inputs = self.safety_wrapper.validate_input(inputs)
           
           # Regular forward pass
           outputs = self.model(inputs, training=training)
           
           # Safety checks after forward pass
           outputs = self.safety_wrapper.validate_output(outputs)
           return outputs
   ```

2. Configure TensorFlow training:
   ```python
   tensorflow_config = {
       "framework": "tensorflow",
       "safety": {
           "enabled": True,
           "checks": {
               "input_validation": True,
               "output_validation": True,
               "gradient_clipping": True
           },
           "metrics": {
               "track_gradients": True,
               "track_activations": True
           }
       },
       "training": {
           "safety_weight": 0.1,
           "gradient_clip_norm": 1.0
       }
   }
   ```

3. Run TensorFlow training:
   ```bash
   python safety_features/scripts/train_model_with_safety.py \
       --model_name SafeTensorFlowModel \
       --config_path tensorflow_config.py \
       --framework tensorflow
   ```

#### 3.2.3 JAX Training

1. Create a JAX model with safety features:
   ```python
   import jax
   import flax.linen as nn
   from safety_features.frameworks.jax import JAXSafetyWrapper

   class SafeJAXModel(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.model = YourJAXModel()
           self.safety_wrapper = JAXSafetyWrapper(config)
           
       def __call__(self, x, training=False):
           # Safety checks before forward pass
           x = self.safety_wrapper.validate_input(x)
           
           # Regular forward pass
           output = self.model(x, training=training)
           
           # Safety checks after forward pass
           output = self.safety_wrapper.validate_output(output)
           return output
   ```

2. Configure JAX training:
   ```python
   jax_config = {
       "framework": "jax",
       "safety": {
           "enabled": True,
           "checks": {
               "input_validation": True,
               "output_validation": True,
               "gradient_clipping": True
           },
           "metrics": {
               "track_gradients": True,
               "track_activations": True
           }
       },
       "training": {
           "safety_weight": 0.1,
           "gradient_clip_norm": 1.0
       }
   }
   ```

3. Run JAX training:
   ```bash
   python safety_features/scripts/train_model_with_safety.py \
       --model_name SafeJAXModel \
       --config_path jax_config.py \
       --framework jax
   ```

### 3.3 Model-Specific Training

#### 3.3.1 Transformer Models (BERT, GPT, etc.)

1. Configure transformer training:
   ```python
   transformer_config = {
       "model": {
           "type": "transformer",
           "architecture": "bert-base-uncased",
           "safety_features": {
               "attention_patterns": True,
               "token_distributions": True,
               "bias_metrics": True
           }
       },
       "training": {
           "epochs": 3,
           "batch_size": 32,
           "learning_rate": 1e-4
       }
   }
   ```

2. Run transformer training:
   ```bash
   python safety_features/scripts/train_model_with_safety.py \
       --model_name SafeTransformerModel \
       --config_path transformer_config.py \
       --model_type transformer
   ```

#### 3.3.2 CNN Models

1. Configure CNN training:
   ```python
   cnn_config = {
       "model": {
           "type": "cnn",
           "architecture": "resnet50",
           "safety_features": {
               "feature_maps": True,
               "activation_patterns": True,
               "gradient_flow": True
           }
       },
       "training": {
           "epochs": 10,
           "batch_size": 64,
           "learning_rate": 1e-3
       }
   }
   ```

2. Run CNN training:
   ```bash
   python safety_features/scripts/train_model_with_safety.py \
       --model_name SafeCNNModel \
       --config_path cnn_config.py \
       --model_type cnn
   ```

#### 3.3.3 RNN/LSTM Models

1. Configure RNN training:
   ```python
   rnn_config = {
       "model": {
           "type": "rnn",
           "architecture": "lstm",
           "safety_features": {
               "hidden_states": True,
               "sequence_patterns": True,
               "memory_usage": True
           }
       },
       "training": {
           "epochs": 5,
           "batch_size": 32,
           "learning_rate": 1e-3
       }
   }
   ```

2. Run RNN training:
   ```bash
   python safety_features/scripts/train_model_with_safety.py \
       --model_name SafeRNNModel \
       --config_path rnn_config.py \
       --model_type rnn
   ```

### 3.4 Monitoring Training Progress

1. View safety metrics during training:
   ```bash
   python safety_features/scripts/monitor_training.py \
       --run_id <training_run_id> \
       --metrics all
   ```

2. Access the safety dashboard:
   ```bash
   python safety_features/scripts/run_dashboard.py
   ```
   Then open your browser to `http://localhost:5001`

3. Export training reports:
   ```bash
   python safety_features/scripts/generate_training_report.py \
       --run_id <training_run_id> \
       --output_path reports/
   ```

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

#### 3.1.14 Additional Model-Specific Examples

##### A. Vision Transformer (ViT) Safety Implementation
```python
class ViTSafetyWrapper(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = ViTModel.from_pretrained(config.model.architecture)
        self.patch_size = config.model.patch_size
        
    def get_safety_features(self):
        return {
            'patch_safety': self.analyze_patches(),
            'spatial_attention': self.analyze_spatial_attention(),
            'image_quality': self.check_image_quality()
        }
        
    def analyze_patches(self):
        """Analyze image patches for safety"""
        return {
            'patch_entropy': self.compute_patch_entropy(),
            'patch_correlation': self.analyze_patch_correlation(),
            'patch_anomalies': self.detect_patch_anomalies()
        }
        
    def analyze_spatial_attention(self):
        """Analyze spatial attention patterns"""
        return {
            'attention_maps': self.generate_attention_maps(),
            'spatial_bias': self.detect_spatial_bias(),
            'region_importance': self.analyze_region_importance()
        }
```

##### B. Graph Neural Network (GNN) Safety Implementation
```python
class GNNSafetyWrapper(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = GNNModel(config)
        
    def get_safety_features(self):
        return {
            'graph_safety': self.analyze_graph_structure(),
            'node_safety': self.analyze_node_features(),
            'edge_safety': self.analyze_edge_weights()
        }
        
    def analyze_graph_structure(self):
        """Analyze graph structure for safety"""
        return {
            'connectivity': self.check_connectivity(),
            'subgraph_patterns': self.analyze_subgraphs(),
            'graph_robustness': self.check_graph_robustness()
        }
```

##### C. Reinforcement Learning Safety Implementation
```python
class RLSafetyWrapper(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = RLModel(config)
        
    def get_safety_features(self):
        return {
            'action_safety': self.analyze_actions(),
            'state_safety': self.analyze_states(),
            'reward_safety': self.analyze_rewards()
        }
        
    def analyze_actions(self):
        """Analyze actions for safety"""
        return {
            'action_distribution': self.check_action_distribution(),
            'action_constraints': self.verify_action_constraints(),
            'action_impact': self.assess_action_impact()
        }
```

#### 3.1.15 Advanced Safety Metrics

##### A. Fairness Metrics
```python
class FairnessMetrics(BaseSafetyMetric):
    def compute(self, model_output, sensitive_attributes):
        """Compute fairness metrics"""
        return {
            'demographic_parity': self.compute_demographic_parity(
                model_output, sensitive_attributes),
            'equal_opportunity': self.compute_equal_opportunity(
                model_output, sensitive_attributes),
            'equalized_odds': self.compute_equalized_odds(
                model_output, sensitive_attributes)
        }
        
    def compute_demographic_parity(self, output, attributes):
        """Compute demographic parity score"""
        return {
            'score': self._calculate_parity_score(output, attributes),
            'bias_metrics': self._compute_bias_metrics(output, attributes)
        }
```

##### B. Robustness Metrics
```python
class RobustnessMetrics(BaseSafetyMetric):
    def compute(self, model_output, perturbations):
        """Compute robustness metrics"""
        return {
            'adversarial_robustness': self.check_adversarial_robustness(
                model_output, perturbations),
            'distribution_shift': self.analyze_distribution_shift(
                model_output),
            'uncertainty_metrics': self.compute_uncertainty(
                model_output)
        }
        
    def check_adversarial_robustness(self, output, perturbations):
        """Check model robustness to adversarial attacks"""
        return {
            'attack_success_rate': self._compute_attack_success(output, perturbations),
            'robustness_score': self._calculate_robustness_score(output, perturbations)
        }
```

##### C. Privacy Metrics
```python
class PrivacyMetrics(BaseSafetyMetric):
    def compute(self, model_output, training_data):
        """Compute privacy metrics"""
        return {
            'membership_inference': self.check_membership_inference(
                model_output, training_data),
            'data_leakage': self.analyze_data_leakage(
                model_output),
            'privacy_score': self.compute_privacy_score(
                model_output)
        }
```

#### 3.1.16 Detailed Dashboard Visualizations

##### A. Real-time Monitoring Dashboard
```python
class SafetyDashboard:
    def __init__(self, config):
        self.config = config
        self.metrics = {}
        self.alerts = []
        
    def update_metrics(self, new_metrics):
        """Update dashboard metrics"""
        self.metrics.update(new_metrics)
        self._check_alerts()
        self._update_visualizations()
        
    def _update_visualizations(self):
        """Update dashboard visualizations"""
        self._update_metric_charts()
        self._update_alert_panel()
        self._update_performance_graphs()
        
    def _update_metric_charts(self):
        """Update metric visualization charts"""
        charts = {
            'safety_scores': self._create_safety_score_chart(),
            'violation_trends': self._create_violation_trend_chart(),
            'performance_impact': self._create_performance_chart()
        }
        return charts
```

##### B. Historical Analysis Dashboard
```python
class HistoricalAnalysisDashboard:
    def __init__(self, config):
        self.config = config
        self.history = {}
        
    def analyze_history(self, time_range):
        """Analyze historical safety data"""
        return {
            'trend_analysis': self._analyze_trends(time_range),
            'violation_patterns': self._analyze_violations(time_range),
            'performance_evolution': self._analyze_performance(time_range)
        }
        
    def _analyze_trends(self, time_range):
        """Analyze safety metric trends"""
        return {
            'metric_trends': self._compute_metric_trends(time_range),
            'correlation_analysis': self._analyze_correlations(time_range),
            'anomaly_detection': self._detect_anomalies(time_range)
        }
```

##### C. Custom Visualization Dashboard
```python
class CustomVisualizationDashboard:
    def __init__(self, config):
        self.config = config
        self.custom_charts = {}
        
    def create_custom_chart(self, chart_config):
        """Create custom visualization chart"""
        return {
            'chart_type': self._determine_chart_type(chart_config),
            'data_processing': self._process_chart_data(chart_config),
            'visualization': self._create_visualization(chart_config)
        }
        
    def _create_visualization(self, config):
        """Create custom visualization"""
        return {
            'plot': self._generate_plot(config),
            'interactivity': self._add_interactivity(config),
            'export_options': self._configure_export(config)
        }
```

#### 3.1.17 Example Dashboard Configuration

```python
dashboard_config = {
    "real_time_monitoring": {
        "enabled": True,
        "update_frequency": "1s",
        "metrics": {
            "safety_scores": {
                "type": "line_chart",
                "refresh_rate": "5s",
                "alerts": {
                    "threshold": 0.8,
                    "notification": "email"
                }
            },
            "violation_trends": {
                "type": "bar_chart",
                "refresh_rate": "1m",
                "categories": ["input", "output", "training"]
            }
        }
    },
    "historical_analysis": {
        "enabled": True,
        "time_ranges": ["1d", "1w", "1m", "1y"],
        "analysis_types": {
            "trend_analysis": True,
            "violation_patterns": True,
            "performance_evolution": True
        }
    },
    "custom_visualizations": {
        "enabled": True,
        "charts": {
            "safety_heatmap": {
                "type": "heatmap",
                "data_source": "safety_metrics",
                "update_frequency": "5m"
            },
            "violation_correlation": {
                "type": "scatter_plot",
                "data_source": "violation_data",
                "update_frequency": "1h"
            }
        }
    }
}
```

[Source: `safety_features/dashboard/config.py`]

#### 3.1.18 Framework-Specific Integration

##### A. PyTorch Integration

###### 1. Basic PyTorch Model with Safety
```python
import torch
import torch.nn as nn
from safety_features.frameworks.pytorch import PyTorchSafetyWrapper

class SafePyTorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = YourPyTorchModel()
        self.safety_wrapper = PyTorchSafetyWrapper(config)
        
    def forward(self, x):
        # Safety checks before forward pass
        x = self.safety_wrapper.validate_input(x)
        
        # Regular forward pass
        output = self.model(x)
        
        # Safety checks after forward pass
        output = self.safety_wrapper.validate_output(output)
        return output
        
    def get_safety_metrics(self):
        return self.safety_wrapper.get_metrics()
```

###### 2. PyTorch Training Loop with Safety
```python
from safety_features.frameworks.pytorch import PyTorchSafetyTrainer

class SafePyTorchTrainer(PyTorchSafetyTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.safety_metrics = {}
        
    def training_step(self, batch, batch_idx):
        # Safety checks on batch
        batch = self.validate_batch(batch)
        
        # Forward pass with safety
        output = self.model(batch)
        
        # Compute loss with safety penalty
        loss = self.compute_loss(output, batch)
        safety_penalty = self.compute_safety_penalty(output)
        total_loss = loss + self.config.safety_weight * safety_penalty
        
        # Update safety metrics
        self.update_safety_metrics(output)
        
        return total_loss
        
    def compute_safety_penalty(self, output):
        """Compute safety violation penalty"""
        return self.model.safety_wrapper.compute_penalty(output)
```

##### B. TensorFlow Integration

###### 1. Basic TensorFlow Model with Safety
```python
import tensorflow as tf
from safety_features.frameworks.tensorflow import TensorFlowSafetyWrapper

class SafeTensorFlowModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.model = YourTensorFlowModel()
        self.safety_wrapper = TensorFlowSafetyWrapper(config)
        
    def call(self, inputs, training=False):
        # Safety checks before forward pass
        inputs = self.safety_wrapper.validate_input(inputs)
        
        # Regular forward pass
        outputs = self.model(inputs, training=training)
        
        # Safety checks after forward pass
        outputs = self.safety_wrapper.validate_output(outputs)
        return outputs
        
    def get_safety_metrics(self):
        return self.safety_wrapper.get_metrics()
```

###### 2. TensorFlow Training with Safety
```python
from safety_features.frameworks.tensorflow import TensorFlowSafetyTrainer

class SafeTensorFlowTrainer(TensorFlowSafetyTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.safety_metrics = {}
        
    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            # Safety checks on batch
            batch = self.validate_batch(batch)
            
            # Forward pass with safety
            output = self.model(batch, training=True)
            
            # Compute loss with safety penalty
            loss = self.compute_loss(output, batch)
            safety_penalty = self.compute_safety_penalty(output)
            total_loss = loss + self.config.safety_weight * safety_penalty
            
        # Update model weights
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update safety metrics
        self.update_safety_metrics(output)
        
        return total_loss
```

##### C. JAX/Flax Integration

###### 1. Basic JAX Model with Safety
```python
import jax
import flax.linen as nn
from safety_features.frameworks.jax import JAXSafetyWrapper

class SafeJAXModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = YourJAXModel()
        self.safety_wrapper = JAXSafetyWrapper(config)
        
    def __call__(self, x, training=False):
        # Safety checks before forward pass
        x = self.safety_wrapper.validate_input(x)
        
        # Regular forward pass
        output = self.model(x, training=training)
        
        # Safety checks after forward pass
        output = self.safety_wrapper.validate_output(output)
        return output
        
    def get_safety_metrics(self):
        return self.safety_wrapper.get_metrics()
```

###### 2. JAX Training with Safety
```python
from safety_features.frameworks.jax import JAXSafetyTrainer

class SafeJAXTrainer(JAXSafetyTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.safety_metrics = {}
        
    @jax.jit
    def train_step(self, state, batch):
        def loss_fn(params):
            # Safety checks on batch
            batch = self.validate_batch(batch)
            
            # Forward pass with safety
            output = self.model.apply(params, batch, training=True)
            
            # Compute loss with safety penalty
            loss = self.compute_loss(output, batch)
            safety_penalty = self.compute_safety_penalty(output)
            return loss + self.config.safety_weight * safety_penalty
            
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        # Update model parameters
        state = state.apply_gradients(grads=grads)
        
        # Update safety metrics
        self.update_safety_metrics(output)
        
        return state, loss
```

##### D. Framework-Specific Configuration

###### 1. PyTorch Configuration
```python
pytorch_config = {
    "framework": "pytorch",
    "safety": {
        "enabled": True,
        "checks": {
            "input_validation": True,
            "output_validation": True,
            "gradient_clipping": True
        },
        "metrics": {
            "track_gradients": True,
            "track_activations": True
        }
    },
    "training": {
        "safety_weight": 0.1,
        "gradient_clip_val": 1.0
    }
}
```

###### 2. TensorFlow Configuration
```python
tensorflow_config = {
    "framework": "tensorflow",
    "safety": {
        "enabled": True,
        "checks": {
            "input_validation": True,
            "output_validation": True,
            "gradient_clipping": True
        },
        "metrics": {
            "track_gradients": True,
            "track_activations": True
        }
    },
    "training": {
        "safety_weight": 0.1,
        "gradient_clip_norm": 1.0
    }
}
```

###### 3. JAX Configuration
```python
jax_config = {
    "framework": "jax",
    "safety": {
        "enabled": True,
        "checks": {
            "input_validation": True,
            "output_validation": True,
            "gradient_clipping": True
        },
        "metrics": {
            "track_gradients": True,
            "track_activations": True
        }
    },
    "training": {
        "safety_weight": 0.1,
        "gradient_clip_norm": 1.0
    }
}
```

[Source: `safety_features/frameworks/config.py`] 