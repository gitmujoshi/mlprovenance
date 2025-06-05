# MNIST Training Report

## Training Run Information
- **Timestamp**: 20250605_094435
- **Model Location**: /Users/mukeshjoshi/gitprojects/mnist_provenance/artifacts/models/20250605_094435
- **Overall Provenance Hash**: `f329e9526fe194d91e6ca54b877de996ebd3bb032098ea24043a4acacbd6a4ae`

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
- **Weights Hash**: `2cb3e549ceaf30c95efea08ea26c77ab4a801ec49d6b400fd95512fd1f14ed4c`

## Training Results
- **Final Accuracy**: 0.9745
- **Final Loss**: 0.0801

### Training Logs
- Epoch 1: accuracy=0.9075, loss=0.3227, val_accuracy=0.9542, val_loss=0.1619
- Epoch 2: accuracy=0.9532, loss=0.1577, val_accuracy=0.9643, val_loss=0.1203
- Epoch 3: accuracy=0.9649, loss=0.1174, val_accuracy=0.9651, val_loss=0.1158
- Epoch 4: accuracy=0.9710, loss=0.0949, val_accuracy=0.9718, val_loss=0.0935
- Epoch 5: accuracy=0.9756, loss=0.0784, val_accuracy=0.9743, val_loss=0.0869

### Training Provenance
- **Training Hash**: `486cf73b7c6b00d6f46167a8197ccd7de62f60faf44f613fa4c959edaca16d45`

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
- **Verification Timestamp**: 1749131081.362886

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
