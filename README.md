# MNIST Provenance Tracking with Differential Privacy

This project demonstrates provenance tracking for a machine learning model trained on the MNIST dataset, now with added differential privacy protection. It includes comprehensive tracking of data, model, and training provenance, along with verification capabilities.

## Features

- **Data Provenance Tracking**
  - Automatic download and verification of MNIST dataset
  - Hash-based tracking of training and test data
  - Data integrity verification through Merkle proofs
  - Support for multiple data mirrors for reliable downloads

- **Model Provenance**
  - Architecture tracking with layer-by-layer verification
  - Weights tracking with hash-based verification
  - Model integrity checks during training
  - Support for model versioning

- **Training Process Monitoring**
  - Real-time tracking of training metrics
  - Privacy budget monitoring
  - Comprehensive logging of training parameters
  - Performance metrics tracking (accuracy, loss)

- **Differential Privacy**
  - Privacy-preserving training using Opacus
  - Configurable privacy budget (epsilon, delta)
  - Gradient clipping and noise addition
  - Privacy budget tracking

- **Verification System**
  - Merkle tree-based verification
  - Proof generation for data, model, and training
  - Comprehensive verification reports
  - Automated integrity checks

- **Reporting**
  - Detailed markdown reports
  - Training statistics and metrics
  - Privacy budget consumption
  - Verification results

## Project Structure

```
mnist_provenance/
├── src/
│   ├── provenance/
│   │   ├── tracker.py      # Provenance tracking implementation
│   │   ├── verifier.py     # Verification system
│   │   ├── merkle_tree.py  # Merkle tree implementation
│   │   └── generate_final_report.py  # Report generation
│   └── training/
│       └── train.py        # Training implementation
├── scripts/
│   ├── setup.sh           # Environment setup script
│   ├── setup_test.py      # Setup verification
│   ├── run_training.sh    # Training execution script
│   └── merkle_proof_demo.py  # Merkle proof demonstration
├── data/                  # MNIST dataset storage
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── artifacts/
│   ├── models/           # Trained model storage
│   └── provenance/       # Provenance reports
└── tests/
    └── test_provenance.py  # Unit tests
```

## Data Management

The project automatically manages the MNIST dataset in the `data/` directory:

### Data Files
- **Training Data:**
  - `train-images-idx3-ubyte.gz`: Compressed training images (60,000 samples)
    - Format: 28x28 grayscale images
    - Size: ~9.5MB compressed
    - Content: Raw pixel values (0-255)
  - `train-labels-idx1-ubyte.gz`: Compressed training labels
    - Format: Single byte per image
    - Size: ~28KB compressed
    - Content: Digit labels (0-9)

- **Test Data:**
  - `t10k-images-idx3-ubyte.gz`: Compressed test images (10,000 samples)
    - Format: 28x28 grayscale images
    - Size: ~1.6MB compressed
    - Content: Raw pixel values (0-255)
  - `t10k-labels-idx1-ubyte.gz`: Compressed test labels
    - Format: Single byte per image
    - Size: ~4.4KB compressed
    - Content: Digit labels (0-9)

### File Format Details
- **Image Files (`*-images-idx3-ubyte.gz`):**
  - Magic number (4 bytes)
  - Number of images (4 bytes)
  - Number of rows (4 bytes)
  - Number of columns (4 bytes)
  - Pixel data (rows × columns × number of images bytes)

- **Label Files (`*-labels-idx1-ubyte.gz`):**
  - Magic number (4 bytes)
  - Number of labels (4 bytes)
  - Label data (1 byte per label)

### Data Processing
- Images are automatically:
  - Normalized to [0, 1] range
  - Reshaped to (N, 1, 28, 28) for PyTorch compatibility
  - Converted to float32 tensors
- Labels are:
  - Converted to long tensors
  - Used as-is (0-9 values)

### Automatic Download
The dataset is automatically downloaded on first run with:
- Multiple mirror support for reliability
- Automatic integrity verification
- Progress tracking during download
- Error handling for failed downloads

### Data Provenance
Each data file is tracked with:
- File hash for integrity verification
- Sample count and basic statistics
- Download source and timestamp
- Processing steps and transformations

### Usage in Training
- Training data (60,000 samples) is used for model training
- Test data (10,000 samples) is used for evaluation
- Data is loaded in batches (default: 64 samples)
- Shuffling is enabled for training data
- Differential privacy is applied during training

## Setup

### Automatic Setup

The easiest way to set up the project is to use the provided setup script:

```bash
./scripts/setup.sh
```

This script will:
1. Create a virtual environment
2. Install all required dependencies
3. Create necessary directories
4. Run a setup test to verify the installation

### Manual Setup

If you prefer to set up manually:

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install PyTorch and torchvision:
```bash
pip install torch==2.7.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Training Script

Run the training script:
```bash
./scripts/run_training.sh
```

This will:
1. Download MNIST dataset if not present
2. Train a model with differential privacy
3. Track all provenance information
4. Generate a detailed report in the artifacts directory

### Training Results

The model achieves:
- Final accuracy: ~88.8%
- Final loss: ~0.416
- Training progress:
  - Epoch 1: 66.1% accuracy
  - Epoch 2: 80.8% accuracy
  - Epoch 3: 84.2% accuracy
  - Epoch 4: 85.3% accuracy
  - Epoch 5: 86.2% accuracy

### Common Warnings and Solutions

During training, you may encounter the following warnings:

1. **NumPy Array Warnings**
   ```
   UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
   ```
   - **Cause**: PyTorch is converting read-only NumPy arrays to tensors
   - **Impact**: No functional impact on training
   - **Solution**: The warning is automatically suppressed after the first occurrence

2. **Secure RNG Warning**
   ```
   UserWarning: Secure RNG turned off. This is perfectly fine for experimentation...
   ```
   - **Cause**: Opacus's secure random number generation is disabled for faster training
   - **Impact**: Slightly reduced security guarantees during experimentation
   - **Solution**: Enable secure mode for production by setting `secure_mode=True` in the PrivacyEngine

3. **Backward Hook Warning**
   ```
   FutureWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes...
   ```
   - **Cause**: PyTorch's internal hook system is being used in a way that will be deprecated
   - **Impact**: No functional impact on training
   - **Solution**: This is an internal PyTorch warning that will be addressed in future versions

These warnings are expected during training and don't affect the model's functionality or privacy guarantees.

### Privacy Parameters

The differential privacy implementation uses:
- Target epsilon (privacy budget): 1.0
- Target delta (failure probability): 1e-5
- Max gradient norm: 1.0

These parameters can be adjusted in `src/training/train.py` to balance privacy and model performance.

### Generating and Verifying Merkle Proofs

You can generate and verify Merkle proofs for any tracked component:

```bash
python scripts/merkle_proof_demo.py
```

The verification system provides:
- Data integrity proofs
- Model architecture verification
- Training process verification
- Overall system integrity checks

## Requirements

- Python 3.8+
- PyTorch 2.7.1
- Torchvision 0.18.1
- Opacus 1.1.3
- NumPy 1.26.4
- Other dependencies as listed in requirements.txt

## License

MIT License 