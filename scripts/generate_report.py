#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
import glob
import json

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from ml_provenance.reporting import BusinessReport

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_latest_provenance_dir():
    """Find the latest provenance directory in artifacts/provenance."""
    provenance_base = Path("artifacts/provenance")
    if not provenance_base.exists():
        return None
    
    # Get all directories matching the timestamp pattern (YYYYMMDD_HHMMSS)
    dirs = [d for d in provenance_base.glob("*") if d.is_dir() and "_" in d.name]
    if not dirs:
        return None
    
    # Sort by directory name (which contains timestamp) in descending order
    return str(sorted(dirs, reverse=True)[0])

def extract_metrics_from_provenance(provenance_file: str) -> dict:
    """Extract training metrics from provenance report."""
    with open(provenance_file, 'r') as f:
        data = json.load(f)
    
    # Extract metrics from the provenance data
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'final_train_acc': 0.0,
        'final_val_acc': 0.0,
        'training_time': 0.0
    }
    
    # Extract metrics from training history
    if 'training_history' in data:
        history = data['training_history']
        for epoch in history:
            metrics['train_loss'].append(epoch.get('train_loss', 0.0))
            metrics['val_loss'].append(epoch.get('val_loss', 0.0))
            metrics['train_acc'].append(epoch.get('train_acc', 0.0))
            metrics['val_acc'].append(epoch.get('val_acc', 0.0))
        
        # Get final metrics
        if metrics['train_acc']:
            metrics['final_train_acc'] = metrics['train_acc'][-1]
        if metrics['val_acc']:
            metrics['final_val_acc'] = metrics['val_acc'][-1]
    
    # Extract training time
    if 'training_time' in data:
        metrics['training_time'] = data['training_time']
    
    return metrics

def extract_privacy_metrics(provenance_file: str) -> dict:
    """Extract privacy metrics from provenance report."""
    with open(provenance_file, 'r') as f:
        data = json.load(f)
    
    privacy_metrics = {
        'privacy_budget': [],
        'target_epsilon': 0.0,
        'final_epsilon': 0.0,
        'delta': 0.0
    }
    
    # Extract privacy metrics
    if 'privacy_metrics' in data:
        privacy = data['privacy_metrics']
        privacy_metrics['privacy_budget'] = privacy.get('privacy_budget', [])
        privacy_metrics['target_epsilon'] = privacy.get('target_epsilon', 0.0)
        privacy_metrics['final_epsilon'] = privacy.get('final_epsilon', 0.0)
        privacy_metrics['delta'] = privacy.get('delta', 0.0)
    
    return privacy_metrics

def extract_verification_results(verification_file: str) -> dict:
    """Extract verification results from verification file."""
    with open(verification_file, 'r') as f:
        data = json.load(f)
    
    verification_results = {}
    
    # Extract verification results
    if 'verification_results' in data:
        for check, result in data['verification_results'].items():
            verification_results[check] = {
                'status': result.get('status', 'unknown'),
                'message': result.get('message', '')
            }
    
    return verification_results

def main():
    """Main function to generate the report."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get the provenance directory from command line argument or find the latest
    if len(sys.argv) == 2:
        provenance_dir = sys.argv[1]
    else:
        provenance_dir = get_latest_provenance_dir()
        if provenance_dir is None:
            logger.error("No provenance directory found in artifacts/provenance")
            sys.exit(1)
        logger.info(f"Using latest provenance directory: {provenance_dir}")
    
    if not os.path.exists(provenance_dir):
        logger.error(f"Provenance directory not found: {provenance_dir}")
        sys.exit(1)
    
    try:
        logger.info(f"Generating report for provenance directory: {provenance_dir}")
        
        # Initialize the report generator
        report_gen = BusinessReport(output_dir="reports")
        
        # Get the required files
        provenance_file = str(Path(provenance_dir) / "provenance_report.json")
        verification_file = str(Path(provenance_dir) / "verification.json")
        
        # Verify required files exist
        for file_path in [provenance_file, verification_file]:
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                sys.exit(1)
        
        # Extract metrics from provenance data
        metrics = extract_metrics_from_provenance(provenance_file)
        privacy_metrics = extract_privacy_metrics(provenance_file)
        verification_results = extract_verification_results(verification_file)
        
        # Save extracted metrics to temporary files
        temp_dir = Path(provenance_dir) / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        metrics_file = str(temp_dir / "metrics.json")
        privacy_file = str(temp_dir / "privacy_metrics.json")
        verification_file = str(temp_dir / "verification_results.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        with open(privacy_file, 'w') as f:
            json.dump(privacy_metrics, f)
        with open(verification_file, 'w') as f:
            json.dump(verification_results, f)
        
        # Generate the report
        report_path = report_gen.generate_report(
            metrics_file=metrics_file,
            privacy_file=privacy_file,
            verification_file=verification_file,
            run_id=Path(provenance_dir).name
        )
        
        # Clean up temporary files
        for file in [metrics_file, privacy_file, verification_file]:
            os.remove(file)
        os.rmdir(temp_dir)
        
        logger.info(f"Report generated successfully at: {report_path}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 