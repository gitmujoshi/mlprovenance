"""
Generate final provenance report for the model.
"""

import json
import os
from datetime import datetime
from pathlib import Path

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
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "verification_report": verification_report,
        "training_config": training_config,
        "verification_status": "PASSED" if verification_report.get("overall_status") else "FAILED"
    }
    
    # Save report in the provenance directory
    report_path = os.path.join(provenance_dir, "provenance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path 