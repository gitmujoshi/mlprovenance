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
    data_path: str,
    training_config: dict,
    output_dir: str = "artifacts/provenance"
) -> str:
    """
    Generate a comprehensive provenance report for the trained model.
    
    Args:
        model_path: Path to the trained model file
        data_path: Path to the training data
        training_config: Training configuration dictionary
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize verifier
    verifier = ProvenanceVerifier()
    
    # Verify model provenance
    model_provenance = verifier.verify_model_provenance(model_path)
    
    # Verify data provenance
    data_provenance = verifier.verify_data_provenance(data_path)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_provenance": model_provenance,
        "data_provenance": data_provenance,
        "training_config": training_config,
        "verification_status": "PASSED" if model_provenance["verified"] and data_provenance["verified"] else "FAILED"
    }
    
    # Save report
    report_path = os.path.join(output_dir, f"provenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path 