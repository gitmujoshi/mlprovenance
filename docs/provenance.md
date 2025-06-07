# Provenance Tracking and Verification

This document describes the provenance tracking and verification system implemented in this project.

## Overview

The system uses a Merkle tree-based approach to track and verify the entire machine learning pipeline, from data to model training. This ensures tamper-evidence and provides a way to verify the integrity of the ML pipeline at any point.

## Merkle Tree Implementation

The Merkle tree is implemented in `src/provenance/merkle_tree.py` and provides the following features:

### Tree Structure
- **Root Node**: Contains the hash of the entire tree
- **Data Node**: Contains hashes of training and test data
- **Model Node**: Contains hashes of model architecture and weights
- **Training Node**: Contains per-epoch nodes with:
  - Model state hash
  - Training metrics (loss, accuracy)
  - Privacy metrics (if applicable)
  - Timestamp and epoch number

### Key Methods
- `add_node(data, node_type)`: Adds a new node to the tree
- `get_proof(node_id)`: Generates a Merkle proof for a node
- `verify_proof(proof, node_id, root_hash)`: Verifies a Merkle proof
- `get_tree_dict()`: Returns the tree structure as a dictionary
- `_hash_data(data)`: Hashes data using SHA-256

### Serialization
- All data is converted to JSON-serializable format before hashing
- Handles PyTorch tensors, NumPy arrays, and native Python types
- Maintains data integrity during serialization

## Verification Logic

The verifier (`src/provenance/verifier.py`) performs comprehensive checks on the ML pipeline:

### Data Verification
1. **Hash Verification**
   - Verifies hash of training and test data
   - Checks data integrity using Merkle proofs
   - Validates data statistics (mean, std, shape)

2. **Metadata Verification**
   - Verifies data source and version
   - Checks data preprocessing steps
   - Validates data splits

### Model Verification
1. **Architecture Verification**
   - Verifies model structure
   - Checks layer configurations
   - Validates parameter counts

2. **Weights Verification**
   - Verifies model weights hash
   - Checks weight statistics
   - Validates weight updates

### Training Verification
1. **Metrics Verification**
   - Verifies training metrics (loss, accuracy)
   - Checks validation metrics
   - Validates metric trends

2. **Privacy Verification**
   - Verifies privacy budget consumption
   - Checks noise addition
   - Validates gradient clipping

3. **Epoch-wise Verification**
   - Verifies each epoch's model state
   - Checks epoch metrics
   - Validates training progression

### Merkle Tree Verification
1. **Tree Structure**
   - Verifies tree structure integrity
   - Checks node relationships
   - Validates hash computations

2. **Proof Verification**
   - Verifies Merkle proofs
   - Checks proof paths
   - Validates root hash

## Verification Process

1. **Initialization**
   - Load provenance data
   - Initialize verifier
   - Set up verification parameters

2. **Data Verification**
   - Verify data hashes
   - Check data statistics
   - Validate metadata

3. **Model Verification**
   - Verify model architecture
   - Check model weights
   - Validate model state

4. **Training Verification**
   - Verify training metrics
   - Check privacy metrics
   - Validate epoch data

5. **Merkle Tree Verification**
   - Verify tree structure
   - Check Merkle proofs
   - Validate root hash

6. **Report Generation**
   - Generate verification report
   - Include detailed results
   - Provide recommendations

## Usage

```python
from provenance.verifier import Verifier

# Initialize verifier
verifier = Verifier(provenance_dir="path/to/provenance")

# Run verification
results = verifier.verify()

# Generate report
verifier.generate_report("verification_report.md")
```

## Best Practices

1. **Regular Verification**
   - Verify after each training run
   - Check before model deployment
   - Validate after data updates

2. **Proof Management**
   - Store proofs securely
   - Verify proofs regularly
   - Update proofs after changes

3. **Error Handling**
   - Handle verification failures
   - Log verification errors
   - Provide clear error messages

4. **Performance**
   - Optimize verification speed
   - Cache verification results
   - Parallelize when possible 