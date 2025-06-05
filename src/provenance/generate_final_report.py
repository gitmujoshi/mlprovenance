import json
import logging
from pathlib import Path
import argparse

def generate_markdown_report(provenance_dir, model_dir):
    """Generate a markdown report from provenance data."""
    logger = logging.getLogger(__name__)
    logger.info("Generating markdown report...")
    
    # Load provenance data
    with open(Path(provenance_dir) / "data.json", "r") as f:
        data = json.load(f)
    
    # Load verification data
    with open(Path(provenance_dir) / "verification.json", "r") as f:
        verification = json.load(f)
    
    # Generate markdown content
    markdown = f"""# MNIST Training Report

## Training Run Information
- **Timestamp**: {data['version']}
- **Model Location**: {model_dir}
- **Overall Provenance Hash**: `{data['hashes']['overall']}`

## Data Statistics
- **Training Samples**: {data['data_provenance']['train']['samples']}
- **Test Samples**: {data['data_provenance']['test']['samples']}
- **Training Data Mean**: {data['data_provenance']['train']['mean']:.4f}
- **Training Data Std**: {data['data_provenance']['train']['std']:.4f}

### Data Provenance
- **Training Data Hash**: `{data['data_provenance']['train']['hash']}`
- **Test Data Hash**: `{data['data_provenance']['test']['hash']}`

## Model Architecture
- **Trainable Parameters**: {data['model_provenance']['parameters']['trainable_params']}
- **Optimizer**: {data['model_provenance']['parameters']['optimizer']['name']}

### Model Provenance
- **Architecture Hash**: `{data['model_provenance']['hashes']['architecture']}`
- **Weights Hash**: `{data['model_provenance']['hashes']['weights']}`

## Training Results
- **Final Accuracy**: {data['training_provenance']['final_metrics']['final_accuracy']:.4f}
- **Final Loss**: {data['training_provenance']['final_metrics']['final_loss']:.4f}

### Training Logs
{chr(10).join(f"- Epoch {i+1}: accuracy={log['accuracy']:.4f}, loss={log['loss']:.4f}, val_accuracy={log['val_accuracy']:.4f}, val_loss={log['val_loss']:.4f}" for i, log in enumerate(data['training_provenance'].get('training_logs', [])))}

### Training Provenance
- **Training Hash**: `{data['training_provenance']['hash']}`

## Privacy Metrics
- **Membership Inference Risk**: {data['training_provenance']['config']['privacy_summary']['membership_inference_risk']:.4f}
- **Model Inversion Risk**: {data['training_provenance']['config']['privacy_summary']['model_inversion_risk']:.4f}
- **Property Inference Risk**: {data['training_provenance']['config']['privacy_summary']['property_inference_risk']:.4f}

## System Information
- **Python Version**: {data['system_info']['python_version']}
- **TensorFlow Version**: {data['system_info']['tensorflow_version']}
- **Platform**: {data['system_info']['platform']['system']} {data['system_info']['platform']['release']}

## Verification Results
- **Overall Status**: {'✅ PASSED' if verification['overall_status'] else '❌ FAILED'}
- **Verification Timestamp**: {verification['verification_timestamp']}

### Data Verification
- **Training Data**: {'✅' if verification['verification_results']['data_verification']['train'] else '❌'}
- **Test Data**: {'✅' if verification['verification_results']['data_verification']['test'] else '❌'}
- **Training Hash**: {'✅' if verification['verification_results']['data_verification']['train_hash'] else '❌'}
- **Test Hash**: {'✅' if verification['verification_results']['data_verification']['test_hash'] else '❌'}

### Model Verification
- **Model Exists**: {'✅' if verification['verification_results']['model_verification']['model_exists'] else '❌'}
- **Architecture Hash Match**: {'✅' if verification['verification_results']['model_verification']['model_hash_match'] else '❌'}
- **Weights Changed During Training**: {'✅' if verification['verification_results']['model_verification']['weights_changed'] else '❌'}
- **Layer Count Match**: {'✅' if verification['verification_results']['model_verification']['architecture_verification']['layer_count_match'] else '❌'}
- **Parameter Count Match**: {'✅' if verification['verification_results']['model_verification']['architecture_verification']['parameter_count_match'] else '❌'}
- **Optimizer Match**: {'✅' if verification['verification_results']['model_verification']['architecture_verification']['optimizer_match'] else '❌'}

### Training Verification
- **Test Accuracy Present**: {'✅' if verification['verification_results']['training_verification']['test_accuracy_present'] else '❌'}
- **Test Loss Present**: {'✅' if verification['verification_results']['training_verification']['test_loss_present'] else '❌'}
- **Privacy Metrics Present**: {'✅' if verification['verification_results']['training_verification']['privacy_metrics_present'] else '❌'}
- **Training Hash Present**: {'✅' if verification['verification_results']['training_verification']['training_hash_present'] else '❌'}

### Hash Verification
- **Model Architecture Hash Match**: {'✅' if verification['verification_results']['hash_verification']['model_architecture_hash_match'] else '❌'}
- **Model Weights Changed During Training**: {'✅' if verification['verification_results']['hash_verification']['model_weights_changed'] else '❌'}
- **Training Hash Match**: {'✅' if verification['verification_results']['hash_verification']['training_hash_match'] else '❌'}
- **Overall Hash Present**: {'✅' if verification['verification_results']['hash_verification']['overall_hash_present'] else '❌'}
"""
    
    # Save markdown report
    report_path = Path(provenance_dir) / "final_report.md"
    with open(report_path, "w") as f:
        f.write(markdown)
    
    logger.info(f"Markdown report generated at {report_path}")
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
    report_path = generate_markdown_report(provenance_dir, model_dir)
    logger.info(f"Final report generated successfully at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final training report")
    parser.add_argument("provenance_dir", help="Directory containing provenance data")
    parser.add_argument("model_dir", help="Directory containing the trained model")
    args = parser.parse_args()
    main(args) 