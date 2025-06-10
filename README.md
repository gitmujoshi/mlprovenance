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
  - Per-epoch Merkle tree updates with:
    - Model state hash
    - Training metrics (loss, accuracy)
    - Privacy metrics (if applicable)
    - Timestamp and epoch number

- **Differential Privacy**
  - Privacy-preserving training using Opacus
  - Configurable privacy budget (epsilon, delta)
  - Gradient clipping and noise addition
  - Privacy budget tracking

- **Verification System**
  - Merkle tree-based verification with:
    - Root hash verification
    - Component-wise hash verification
    - Proof generation for data, model, and training
    - Automated integrity checks
  - Comprehensive verification reports including:
    - Data verification (hash match, statistics, metadata)
    - Model verification (architecture, weights)
    - Training verification (metrics, logs)
    - Privacy verification (budget consumption)
  - Merkle tree structure:
    ```
    Root Hash
    ├── Data Node
    │   ├── Training Data Hash
    │   └── Test Data Hash
    ├── Model Node
    │   ├── Architecture Hash
    │   └── Weights Hash
    └── Training Node
        ├── Epoch 1 Node
        │   ├── Model State Hash
        │   ├── Metrics Hash
        │   └── Privacy Metrics Hash
        ├── Epoch 2 Node
        │   ├── Model State Hash
        │   ├── Metrics Hash
        │   └── Privacy Metrics Hash
        └── ... (subsequent epochs)
    ```

- **Reporting**
  - Detailed markdown reports
  - Training statistics and metrics
  - Privacy budget consumption
  - Verification results
  - Merkle tree visualization

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
├── docs/
│   ├── technical_documentation.md  # Technical documentation
│   └── user_documentation.md      # User documentation
├── data/                  # MNIST dataset storage
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── artifacts/
│   ├── models/           # Trained model storage
│   └── provenance/       # Provenance reports
├── tests/                # Unit tests
├── setup.py             # Package setup configuration
├── pyproject.toml       # Build system configuration
└── MANIFEST.in          # Package manifest
```

## Installation

### As a Package

The project can be installed as a Python package:

```bash
# Clone the repository
git clone https://github.com/yourusername/mnist_provenance.git
cd mnist_provenance

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows

# Install the package in development mode
pip install -e .
```

### Package Dependencies

The package requires the following dependencies:
- numpy>=1.19.0
- pandas>=1.2.0
- torch>=1.7.0
- tensorflow>=2.4.0
- scikit-learn>=0.24.0
- mlflow>=1.20.0
- blake3>=0.3.0

Development dependencies:
- pytest>=6.0
- black>=21.0
- isort>=5.0
- flake8>=3.9
- mypy>=0.910

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

## Setup Instructions
1. Run the setup script:
   ```sh
   bash scripts/setup.sh
   ```
2. **After setup, activate the virtual environment before running any Python scripts:**
   ```sh
   source venv/bin/activate
   ```
   Or, use the venv's Python directly:
   ```sh
   ./venv/bin/python <your_script.py>
   ```

## Usage

### Running the Training Script

Run the training script:
```bash
./scripts/run_training.sh
```

This will:
1. Install the package in development mode
2. Download MNIST dataset if not present
3. Train a model with differential privacy
4. Track all provenance information
5. Generate a detailed report in the artifacts directory

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

## Usage
- **Tip:** Always make sure your virtual environment is activated before running training or verification scripts. If not, use `./venv/bin/python` to ensure the correct environment is used. 