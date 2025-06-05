# MNIST Provenance Tracking

This project demonstrates provenance tracking for a machine learning model trained on the MNIST dataset. It includes comprehensive tracking of data, model, and training provenance, along with verification capabilities.

## Features

- Data provenance tracking
- Model architecture and weights tracking
- Training process monitoring
- Comprehensive verification system
- Detailed reporting with markdown output
- Merkle tree-based verification and proof generation

## Project Structure

```
mnist_provenance/
├── src/
│   ├── provenance/
│   │   ├── tracker.py
│   │   ├── verifier.py
│   │   ├── merkle_tree.py
│   │   └── generate_final_report.py
│   └── training/
│       └── train.py
├── scripts/
│   ├── run_training.sh
│   └── merkle_proof_demo.py
├── artifacts/
│   ├── models/
│   └── provenance/
└── tests/
    └── test_provenance.py
```

## Setup

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

3. Install dependencies:
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
1. Train a model on the MNIST dataset
2. Track all provenance information
3. Generate a detailed report in the artifacts directory, including training metrics, provenance hashes, Merkle proofs, and verification results

### Generating and Verifying Merkle Proofs

You can generate and verify Merkle proofs for any tracked component (data, model, or training) using the provided script:

```bash
python scripts/merkle_proof_demo.py
```

This script will:
- Find the latest provenance directory
- Load the provenance data
- Rebuild the Merkle tree for that run
- Generate a Merkle proof for the selected component
- Verify the proof against the Merkle root

You can change the component type in the script to `'data'`, `'model'`, or `'training'` to generate and verify proofs for different components.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- pytest (for testing)

## License

MIT License 