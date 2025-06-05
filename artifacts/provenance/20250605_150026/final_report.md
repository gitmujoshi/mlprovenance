# MNIST Training Report

## Training Run Information
- **Timestamp**: 20250605_150026
- **Model Location**: /Users/mukeshjoshi/gitprojects/mnist_provenance/artifacts/models/20250605_150026
- **Overall Provenance Hash**: `303672e4883505838c86b595ca0870bd78bde707217875a44c63f4b5aeb2c2ef`

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
- **Weights Hash**: `fef1dbaa2b644982d9576f4f0cb4b9c3c1fc8ae191788f91f7ff7fb5bc6eb4fc`

## Training Results
- **Final Accuracy**: 0.9736
- **Final Loss**: 0.0839

### Training Logs
- Epoch 1: accuracy=0.9045, loss=0.3262, val_accuracy=0.9549, val_loss=0.1538
- Epoch 2: accuracy=0.9526, loss=0.1583, val_accuracy=0.9660, val_loss=0.1130
- Epoch 3: accuracy=0.9643, loss=0.1175, val_accuracy=0.9696, val_loss=0.1028
- Epoch 4: accuracy=0.9703, loss=0.0972, val_accuracy=0.9714, val_loss=0.0910
- Epoch 5: accuracy=0.9747, loss=0.0815, val_accuracy=0.9723, val_loss=0.0873

### Training Provenance
- **Training Hash**: `3758bafac59b6fe8bad56de406453a453699c3d96132486223fac75f9d5f2b80`

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
- **Verification Timestamp**: 1749150031.77704

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
