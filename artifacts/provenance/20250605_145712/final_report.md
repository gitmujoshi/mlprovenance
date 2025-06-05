# MNIST Training Report

## Training Run Information
- **Timestamp**: 20250605_145712
- **Model Location**: /Users/mukeshjoshi/gitprojects/mnist_provenance/artifacts/models/20250605_145712
- **Overall Provenance Hash**: `d3a2cf3ece83ef7583b3c7a9a1e7cce2cfa17c641600b89ec0e9c3c7f1f06af1`

## Data Statistics
- **Training Samples**: 60000
- **Test Samples**: 10000
- **Training Data Mean**: 0.1307
- **Training Data Std**: 0.3081

### Data Provenance
- **Training Data Hash**: `a32bf55afb4a5c9cd4224cab3fb024d518fc9e9254ca071befcf5a828eff8c73`
- **Test Data Hash**: `66a1790b97c8e5364c6b703f03902cb548dcb6c22113b17ca209b037fdaf5549`

## Model Architecture
- **Trainable Parameters**: 101770
- **Optimizer**: Adam

### Model Provenance
- **Architecture Hash**: `530f7474f834a993a15920e4e7811f97ad1311ab0485519af458a926ec51108e`
- **Weights Hash**: `6e00ba1092cd63890c49cdb79586a1b67b8d628e4cdc23f12d8618aa0f77f6ae`

## Training Results
- **Final Accuracy**: 0.9744
- **Final Loss**: 0.0825

### Training Logs
- Epoch 1: accuracy=0.9020, loss=0.3327, val_accuracy=0.9532, val_loss=0.1653
- Epoch 2: accuracy=0.9516, loss=0.1631, val_accuracy=0.9661, val_loss=0.1181
- Epoch 3: accuracy=0.9643, loss=0.1214, val_accuracy=0.9702, val_loss=0.1004
- Epoch 4: accuracy=0.9700, loss=0.0982, val_accuracy=0.9709, val_loss=0.0983
- Epoch 5: accuracy=0.9750, loss=0.0821, val_accuracy=0.9752, val_loss=0.0857

### Training Provenance
- **Training Hash**: `c156a1442a20c0a8abb706fd53fcbf6906bbbe7bc6ee1279e34c3e1ae83f800a`

## Privacy Metrics
- **Membership Inference Risk**: 0.1500
- **Model Inversion Risk**: 0.1000
- **Property Inference Risk**: 0.0500

## System Information
- **Python Version**: 3.11.12 (main, Apr  8 2025, 14:15:29) [Clang 17.0.0 (clang-1700.0.13.3)]
- **TensorFlow Version**: 2.19.0
- **Platform**: Darwin 24.5.0

## Verification Results
- **Overall Status**: ✅ PASSED
- **Verification Timestamp**: 1749149838.452775

### Data Verification
- **Training Data**: ✅
- **Test Data**: ✅
- **Training Hash**: ✅
- **Test Hash**: ✅

### Model Verification
- **Model Exists**: ✅
- **Architecture Hash Match**: ✅
- **Weights Changed During Training**: ✅
- **Layer Count Match**: ✅
- **Parameter Count Match**: ✅
- **Optimizer Match**: ✅

### Training Verification
- **Test Accuracy Present**: ✅
- **Test Loss Present**: ✅
- **Privacy Metrics Present**: ✅
- **Training Hash Present**: ✅

### Hash Verification
- **Model Architecture Hash Match**: ✅
- **Model Weights Changed During Training**: ✅
- **Training Hash Match**: ✅
- **Overall Hash Present**: ✅
