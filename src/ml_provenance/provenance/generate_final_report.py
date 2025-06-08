import json
import logging
from pathlib import Path
import argparse
import os
from typing import Dict, Any
from datetime import datetime
from src.provenance.verifier import ProvenanceVerifier

def generate_final_report(provenance_dir, model_path, output_path=None):
    """Generate a final report for the training run."""
    verifier = ProvenanceVerifier(provenance_dir)
    verification_results = verifier.generate_verification_report(model_path)
    
    # Load provenance data
    with open(provenance_dir / "provenance.json", "r") as f:
        data = json.load(f)
    
    # Generate report content
    report_content = []
    report_content.append("# Training Report\n")
    
    # Initial State
    report_content.append("## Initial State\n")
    report_content.append("### Data")
    report_content.append(f"- Train Data Hash: {data['data_provenance']['train']['hash']}")
    report_content.append(f"- Test Data Hash: {data['data_provenance']['test']['hash']}")
    report_content.append(f"- Timestamp: {data['version']}\n")
    
    # Training Summary
    report_content.append("## Training Summary\n")
    report_content.append("### Parameters")
    report_content.append(f"- Epochs: {data['training_provenance']['epochs']}")
    report_content.append(f"- Batch Size: {data['training_provenance']['batch_size']}")
    report_content.append(f"- Learning Rate: {data['training_provenance']['learning_rate']}\n")
    
    # Training Logs
    report_content.append("### Training Logs")
    for log in data['training_provenance']['training_logs']:
        report_content.append(f"Epoch {log['epoch']}:")
        report_content.append(f"- Loss: {log['loss']:.4f}")
        report_content.append(f"- Train Accuracy: {log['train_accuracy']:.2f}%")
        report_content.append(f"- Test Accuracy: {log['test_accuracy']:.2f}%\n")
    
    # Privacy Metrics
    report_content.append("## Privacy Metrics\n")
    privacy_metrics = data['training_provenance'].get('privacy_metrics', {})
    report_content.append("### Configuration")
    report_content.append(f"- Target Epsilon: {privacy_metrics.get('target_epsilon', 'Not set')}")
    report_content.append(f"- Target Delta: {privacy_metrics.get('target_delta', 'Not set')}")
    report_content.append(f"- Noise Multiplier: {privacy_metrics.get('noise_multiplier', 'Not set')}\n")
    
    # Privacy Guarantees
    report_content.append("### Privacy Guarantees")
    report_content.append(f"- Achieved Epsilon: {privacy_metrics.get('achieved_epsilon', 'Not calculated')}")
    report_content.append(f"- Achieved Delta: {privacy_metrics.get('achieved_delta', 'Not calculated')}")
    report_content.append(f"- Privacy Budget Used: {privacy_metrics.get('privacy_budget_used', 'Not calculated')}%\n")
    
    # Final State
    report_content.append("## Final State\n")
    if 'final_metrics' in data['training_provenance']:
        report_content.append(f"- Final Loss: {data['training_provenance']['final_metrics']['loss']:.4f}")
        report_content.append(f"- Final Train Accuracy: {data['training_provenance']['final_metrics']['train_accuracy']:.2f}%")
        report_content.append(f"- Final Test Accuracy: {data['training_provenance']['final_metrics']['test_accuracy']:.2f}%\n")
    else:
        report_content.append("No final metrics available.\n")
    
    # Verification Results
    report_content.append("## Verification Results\n")
    report_content.append("### Component Verification")
    report_content.append(f"- Data: {'✅' if verification_results['verification_results']['data_verification']['merkle_verified'] else '❌'}")
    report_content.append(f"- Model: {'✅' if verification_results['verification_results']['model_verification']['model_hash_match'] else '❌'}")
    report_content.append(f"- Training: {'✅' if verification_results['verification_results']['training_verification']['final_metrics_present'] else '❌'}")
    report_content.append(f"- Privacy: {'✅' if verification_results['verification_results']['hash_verification']['privacy_hash_match'] else '❌'}\n")
    
    # Merkle Tree Verification
    report_content.append("### Merkle Tree Verification")
    report_content.append(f"- Tree Structure: {'✅' if verification_results['overall_verification']['tree_structure_valid'] else '❌'}")
    report_content.append(f"- Hash Chain: {'✅' if verification_results['overall_verification']['hash_chain_valid'] else '❌'}")
    report_content.append(f"- Timestamp Chain: {'✅' if verification_results['overall_verification']['timestamp_chain_valid'] else '❌'}\n")
    
    # Overall Status
    report_content.append("### Overall Status")
    report_content.append(f"- Verification Status: {'✅ SUCCESS' if verification_results['overall_status'] else '❌ FAILURE'}\n")
    
    # Save report
    report_path = output_path if output_path is not None else provenance_dir / "final_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_content))
    
    print(f"Final report generated at: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate final training report")
    parser.add_argument("--provenance-dir", type=str, required=True, help="Directory containing provenance data")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    args = parser.parse_args()
    
    generate_final_report(Path(args.provenance_dir), args.model_path)

if __name__ == "__main__":
    main() 