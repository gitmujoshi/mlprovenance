"""
Generate final provenance report for the model.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import re

from ml_provenance.provenance.verifier import ProvenanceVerifier
from ml_provenance.provenance.tracker import ProvenanceTracker

def generate_final_report(
    model_path: str,
    provenance_dir: str,
    training_config: dict,
    output_dir: str = None
) -> str:
    """
    Generate a comprehensive provenance report for the trained model.
    
    Args:
        model_path: Path to the trained model file
        provenance_dir: Directory containing provenance data (provenance.json)
        training_config: Training configuration dictionary
        output_dir: Directory to save the report (defaults to provenance_dir)
        
    Returns:
        Path to the generated report
    """
    # Use provenance_dir as output_dir if not specified
    if output_dir is None:
        output_dir = provenance_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize verifier with provenance directory
    verifier = ProvenanceVerifier(provenance_dir)
    verification_report = verifier.generate_verification_report(model_path)
    
    # Look for existing detailed provenance data
    existing_provenance = {}
    provenance_file = os.path.join(provenance_dir, "provenance_report.json")
    if os.path.exists(provenance_file):
        try:
            with open(provenance_file, 'r') as f:
                existing_provenance = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read existing provenance file: {e}")
    
    # Look for Merkle tree information
    merkle_info = {}
    merkle_files = list(Path(provenance_dir).glob("merkle_tree_*.json"))
    if merkle_files:
        # Get the most recent Merkle tree file
        latest_merkle_file = max(merkle_files, key=lambda x: x.stat().st_mtime)
        try:
            # Extract just the root hash from the beginning of the file
            with open(latest_merkle_file, 'r') as f:
                # Read first 1000 characters to find the root hash
                content = f.read(1000)
                # Look for the root hash pattern
                root_match = re.search(r'"hash":\s*"([a-f0-9]{64})"', content)
                if root_match:
                    root_hash = root_match.group(1)
                    merkle_info = {
                        "root_hash": root_hash,
                        "tree_file": latest_merkle_file.name,
                        "algorithm": training_config.get("hash_algorithm", "blake3")
                    }
                else:
                    print(f"Warning: Could not find root hash in {latest_merkle_file}")
        except Exception as e:
            print(f"Warning: Could not read Merkle tree file {latest_merkle_file}: {e}")
    
    # Merge existing provenance data with new verification data
    report = existing_provenance.copy()  # Start with existing detailed data
    
    # Add/update verification information
    report.update({
        "verification_report": verification_report,
        "verification_status": "PASSED" if verification_report.get("verification_status") == "success" else "FAILED"
    })
    
    # Add Merkle tree information if available (don't overwrite if already present)
    if merkle_info and "merkle_tree" not in report:
        report["merkle_tree"] = merkle_info
    elif merkle_info:
        # Update existing merkle info with new data
        report["merkle_tree"].update(merkle_info)
    
    # Save merged report in the provenance directory
    report_path = os.path.join(provenance_dir, "provenance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path 