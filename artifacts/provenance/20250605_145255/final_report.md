# MNIST Training Report

## Training Run Information
- **Timestamp**: 20250605_145255
- **Model Location**: /Users/mukeshjoshi/gitprojects/mnist_provenance/artifacts/models/20250605_145255
- **Overall Provenance Hash**: `a09a4167765fe75b558c9bc716e8f54790152fc22b425960fbb1dd669ee159e9`

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
- **Weights Hash**: `9474eec250568fc97f61944f3586364e93aeb03af407cde4455a461fec605d8b`

## Training Results
- **Final Accuracy**: 0.9730
- **Final Loss**: 0.0854

### Training Logs
- Epoch 1: accuracy=0.9027, loss=0.3322, val_accuracy=0.9559, val_loss=0.1586
- Epoch 2: accuracy=0.9517, loss=0.1615, val_accuracy=0.9632, val_loss=0.1235
- Epoch 3: accuracy=0.9642, loss=0.1206, val_accuracy=0.9672, val_loss=0.1102
- Epoch 4: accuracy=0.9701, loss=0.0973, val_accuracy=0.9725, val_loss=0.0947
- Epoch 5: accuracy=0.9744, loss=0.0828, val_accuracy=0.9722, val_loss=0.0941

### Training Provenance
- **Training Hash**: `571901528bd41fd45a79757f21b208ed77dafbe17621e1fc937b2f2c1f3f5dff`

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
- **Verification Timestamp**: 1749149580.87326

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
