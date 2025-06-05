# MNIST Provenance Tracking

This project demonstrates provenance tracking for a machine learning model trained on the MNIST dataset. It includes comprehensive tracking of data, model, and training provenance, along with verification capabilities.

## Features

- Data provenance tracking
- Model architecture and weights tracking
- Training process monitoring
- Comprehensive verification system
- Detailed reporting with markdown output

## Project Structure

```
mnist_provenance/
├── src/
│   ├── provenance/
│   │   ├── tracker.py
│   │   ├── verifier.py
│   │   └── generate_final_report.py
│   └── training/
│       └── train.py
├── scripts/
│   └── run_training.sh
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

Run the training script:
```bash
./scripts/run_training.sh
```

This will:
1. Train a model on the MNIST dataset
2. Track all provenance information
3. Generate a detailed report in the artifacts directory

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- pytest (for testing)

## License

MIT License 