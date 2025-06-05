# MNIST Training Report

## Training Run Information
- **Timestamp**: 20250605_094752
- **Model Location**: /Users/mukeshjoshi/gitprojects/mnist_provenance/artifacts/models/20250605_094752
- **Overall Provenance Hash**: `0f713b010ebe57cb80bcc07e735d1dd75eb31a0f0eba02a7d36c3c50041c0a69`

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
- **Weights Hash**: `820ed3a8ac81b2e0d12af70dd9db758b7b18065924acf30e71ba715b432848fc`

## Training Results
- **Final Accuracy**: 0.9743
- **Final Loss**: 0.0858

### Training Logs
- Epoch 1: accuracy=0.9037, loss=0.3303, val_accuracy=0.9567, val_loss=0.1551
- Epoch 2: accuracy=0.9538, loss=0.1587, val_accuracy=0.9649, val_loss=0.1184
- Epoch 3: accuracy=0.9644, loss=0.1177, val_accuracy=0.9682, val_loss=0.1038
- Epoch 4: accuracy=0.9701, loss=0.0969, val_accuracy=0.9740, val_loss=0.0893
- Epoch 5: accuracy=0.9755, loss=0.0806, val_accuracy=0.9736, val_loss=0.0857

### Training Provenance
- **Training Hash**: `d7b2b8892deec9bd3f0aa841551ce5784c4ed02400910608b04d8d538886b0c9`

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
- **Verification Timestamp**: 1749131277.897962

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
