#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from ml_provenance.provenance.tracker import ProvenanceTracker
from ml_provenance.provenance.merkle_tree import MerkleTree

def get_latest_provenance_dir(base_dir="artifacts/provenance"):
    """Get the most recent provenance directory."""
    dirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No provenance directories found.")
    return str(sorted(dirs)[-1])

def main(component_type="data"):
    # Find the latest provenance directory
    provenance_dir = get_latest_provenance_dir()
    print(f"Using provenance directory: {provenance_dir}")

    # Load provenance data
    with open(os.path.join(provenance_dir, "data.json"), "r") as f:
        provenance_data = json.load(f)

    # Initialize the tracker and rebuild the Merkle tree
    tracker = ProvenanceTracker()
    tracker.merkle_tree.track_training_run(
        provenance_data["data_provenance"],
        provenance_data["model_provenance"],
        provenance_data["training_provenance"]
    )

    # Select the component
    component_map = {
        "data": provenance_data["data_provenance"],
        "model": provenance_data["model_provenance"],
        "training": provenance_data["training_provenance"]
    }
    if component_type not in component_map:
        raise ValueError(f"Invalid component_type: {component_type}")

    component_data = component_map[component_type]

    # Generate Merkle proof
    proof = tracker.get_provenance_proof(component_type, component_data)
    print(f"\nMerkle proof for {component_type} provenance:")
    print(json.dumps(proof, indent=2))

    # Verify the proof
    is_valid = tracker.merkle_tree.tree.verify_data(
        {"type": component_type, "content": component_data}, proof
    )
    print(f"\nVerification result for {component_type} provenance: {'SUCCESS' if is_valid else 'FAILURE'}")

if __name__ == "__main__":
    # Change 'data' to 'model' or 'training' to test other components
    main(component_type="data") 