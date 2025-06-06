import json
import logging
from pathlib import Path
import argparse
import os

def generate_final_report(provenance_dir, model_path, verification_report=None, merkle_proofs=None):
    """
    Generate a detailed markdown report for training, provenance, and Merkle verification.
    """
    provenance_dir = Path(provenance_dir)
    report_path = provenance_dir / "final_report.md"
    with open(provenance_dir / "data.json", "r") as f:
        provenance_data = json.load(f)

    # Section 1: Training Summary
    training = provenance_data["training_provenance"]
    training_logs = training.get("training_logs", [])
    final_metrics = training.get("final_metrics", {})
    training_section = [
        "# Training Summary",
        "",
        f"**Epochs:** {training.get('config', {}).get('epochs', 'N/A')}",
        f"**Batch Size:** {training.get('config', {}).get('batch_size', 'N/A')}",
        f"**Validation Split:** {training.get('config', {}).get('validation_split', 'N/A')}",
        f"**Final Accuracy:** {final_metrics.get('final_accuracy', 'N/A')}",
        f"**Final Loss:** {final_metrics.get('final_loss', 'N/A')}",
        "",
        "## Training Logs",
        "| Epoch | Accuracy | Loss | Val Accuracy | Val Loss |",
        "|-------|----------|------|--------------|----------|",
    ]
    for log in training_logs:
        training_section.append(f"| {log['epoch']} | {log['accuracy']} | {log['loss']} | {log['val_accuracy']} | {log['val_loss']} |")

    # Section 2: Privacy Metrics
    privacy_config = training.get('config', {}).get('privacy_parameters', {})
    privacy_metrics = training.get('privacy_metrics', {})
    privacy_section = [
        "# Privacy Metrics",
        "",
        "## Privacy Configuration",
        f"- Target Epsilon (Privacy Budget): {privacy_config.get('target_epsilon', 'N/A')}",
        f"- Target Delta (Failure Probability): {privacy_config.get('target_delta', 'N/A')}",
        f"- Max Gradient Norm: {privacy_config.get('max_grad_norm', 'N/A')}",
        "",
        "## Privacy Guarantees",
        f"- Achieved Epsilon: {privacy_metrics.get('achieved_epsilon', 'N/A')}",
        f"- Achieved Delta: {privacy_metrics.get('achieved_delta', 'N/A')}",
        f"- Privacy Budget Used: {privacy_metrics.get('privacy_budget_used', 'N/A')}%",
        "",
        "## Privacy Implementation Details",
        "- Differential Privacy: Enabled via Opacus",
        "- Noise Addition: Gaussian noise with clipping",
        "- Secure RNG: " + ("Enabled" if privacy_metrics.get('secure_rng_enabled', False) else "Disabled (experimental mode)"),
        "",
        "## Privacy Impact",
        "- Model Performance Impact: " + privacy_metrics.get('performance_impact', 'N/A'),
        "- Privacy-Utility Trade-off: " + privacy_metrics.get('privacy_utility_tradeoff', 'N/A'),
    ]

    # Section 3: Provenance Details
    hashes = provenance_data.get("hashes", {})
    provenance_section = [
        "# Provenance Details",
        "",
        "## Data Provenance",
        f"- Train Hash: `{provenance_data['data_provenance'].get('train', {}).get('hash', 'N/A')}`",
        f"- Test Hash: `{provenance_data['data_provenance'].get('test', {}).get('hash', 'N/A')}`",
        "",
        "## Model Provenance",
        f"- Architecture Hash: `{provenance_data['model_provenance']['hashes'].get('architecture', 'N/A')}`",
        f"- Weights Hash: `{provenance_data['model_provenance']['hashes'].get('weights', 'N/A')}`",
        "",
        "## Training Provenance",
        f"- Training Hash: `{provenance_data['training_provenance'].get('hash', 'N/A')}`",
        "",
        "## Overall Hashes",
    ]
    for k, v in hashes.items():
        provenance_section.append(f"- {k}: `{v}`")

    # Section 4: Merkle Tree and Proofs
    merkle_section = [
        "# Merkle Tree & Verification",
        "",
        f"**Merkle Root Hash:** `{hashes.get('overall', 'N/A')}`",
        "",
        "## Merkle Proofs",
    ]
    if merkle_proofs:
        for comp, proof in merkle_proofs.items():
            merkle_section.append(f"### Proof for {comp.capitalize()} Provenance:")
            merkle_section.append("```")
            merkle_section.append(json.dumps(proof, indent=2))
            merkle_section.append("```")
    else:
        merkle_section.append("(Proofs not generated in this run)")

    # Section 5: Verification Results
    verification_section = [
        "# Verification Results",
        "",
    ]
    if verification_report:
        verification_section.append("## Overall Status: " + ("✅ SUCCESS" if verification_report.get("overall_status") else "❌ FAILURE"))
        verification_section.append("")
        for key, results in verification_report.get("verification_results", {}).items():
            verification_section.append(f"### {key.replace('_', ' ').title()}")
            verification_section.append("```")
            verification_section.append(json.dumps(results, indent=2))
            verification_section.append("```")
    else:
        verification_section.append("(Verification not performed in this run)")

    # Combine all sections
    report = "\n".join(training_section + [""] + privacy_section + [""] + provenance_section + [""] + merkle_section + [""] + verification_section)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"INFO:src.provenance.generate_final_report:Markdown report generated at {report_path}")
    return report_path

def main(args):
    """Main function to generate the final report."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    provenance_dir = Path(args.provenance_dir)
    model_dir = Path(args.model_dir)
    
    if not provenance_dir.exists():
        raise ValueError(f"Provenance directory not found: {provenance_dir}")
    
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Generate markdown report
    report_path = generate_final_report(provenance_dir, model_dir)
    logger.info(f"Final report generated successfully at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final training report")
    parser.add_argument("provenance_dir", help="Directory containing provenance data")
    parser.add_argument("model_dir", help="Directory containing the trained model")
    args = parser.parse_args()
    main(args) 