import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_provenance_dirs() -> List[str]:
    """Get all provenance directories."""
    provenance_dir = Path("artifacts/provenance")
    if not provenance_dir.exists():
        logger.error("Provenance directory not found")
        return []
    return [d.name for d in provenance_dir.iterdir() if d.is_dir()]

def extract_run_info(run_id: str) -> Dict[str, Any]:
    """Extract information from a run's provenance files."""
    run_info = {
        "data": {},
        "model": {},
        "training": {
            "epochs": 0,
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "final_train_acc": 0.0,
            "final_val_acc": 0.0,
            "training_time": 0.0
        },
        "privacy": {
            "target_epsilon": 0.0,
            "final_epsilon": 0.0,
            "delta": 0.0,
            "privacy_budget": []
        },
        "verification": {
            "data_verification": {},
            "model_verification": {},
            "training_verification": {},
            "privacy_verification": {},
            "overall_status": "FAILED"
        }
    }
    
    # Read provenance.log
    log_path = Path(f"artifacts/provenance/{run_id}/provenance.log")
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if "Dataset:" in line:
                    run_info["data"]["dataset"] = line.split("Dataset:")[1].strip()
                elif "Training samples:" in line:
                    run_info["data"]["train_samples"] = int(line.split(":")[1].strip())
                elif "Test samples:" in line:
                    run_info["data"]["test_samples"] = int(line.split(":")[1].strip())
                elif "Training mean:" in line:
                    run_info["data"]["train_mean"] = float(line.split(":")[1].strip())
                elif "Training std:" in line:
                    run_info["data"]["train_std"] = float(line.split(":")[1].strip())
                elif "Test mean:" in line:
                    run_info["data"]["test_mean"] = float(line.split(":")[1].strip())
                elif "Test std:" in line:
                    run_info["data"]["test_std"] = float(line.split(":")[1].strip())
                elif "Model architecture:" in line:
                    run_info["model"]["architecture"] = line.split(":")[1].strip()
                elif "Total parameters:" in line:
                    run_info["model"]["total_parameters"] = int(line.split(":")[1].strip())
                elif "Number of layers:" in line:
                    run_info["model"]["num_layers"] = int(line.split(":")[1].strip())
                elif "Layer" in line and ":" in line:
                    if "layers" not in run_info["model"]:
                        run_info["model"]["layers"] = []
                    layer_info = line.split(":")[1].strip()
                    if "(" in layer_info and ")" in layer_info:
                        try:
                            name, layer_type = layer_info.split("(")
                            run_info["model"]["layers"].append({
                                "name": name.strip(),
                                "type": layer_type.strip(")")
                            })
                        except Exception as e:
                            logger.warning(f"Could not parse layer info: {layer_info} ({e})")
                    else:
                        logger.warning(f"Skipping malformed layer info: {layer_info}")

    # Read provenance_report.json
    report_path = Path(f"artifacts/provenance/{run_id}/provenance_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report_data = json.load(f)
            
            # Extract training history
            if "training_history" in report_data:
                history = report_data["training_history"]
                run_info["training"]["epochs"] = len(history)
                for epoch in history:
                    run_info["training"]["train_loss"].append(epoch["train_loss"])
                    run_info["training"]["val_loss"].append(epoch["val_loss"])
                    run_info["training"]["train_acc"].append(epoch["train_acc"])
                    run_info["training"]["val_acc"].append(epoch["val_acc"])
                
                if history:
                    run_info["training"]["final_train_acc"] = history[-1]["train_acc"]
                    run_info["training"]["final_val_acc"] = history[-1]["val_acc"]
            
            # Extract privacy metrics
            if "privacy_metrics" in report_data:
                privacy = report_data["privacy_metrics"]
                run_info["privacy"]["target_epsilon"] = privacy.get("target_epsilon", 0.0)
                run_info["privacy"]["final_epsilon"] = privacy.get("final_epsilon", 0.0)
                run_info["privacy"]["delta"] = privacy.get("delta", 0.0)
                run_info["privacy"]["privacy_budget"] = privacy.get("privacy_budget", [])

    # Read verification.json
    verification_path = Path(f"artifacts/provenance/{run_id}/verification.json")
    if verification_path.exists():
        with open(verification_path) as f:
            verification_data = json.load(f)
            
            # Extract verification results
            for key in ["data_verification", "model_verification", "training_verification", "privacy_verification"]:
                if key in verification_data:
                    run_info["verification"][key] = verification_data[key]
            
            # Set overall status based on verification results
            all_verified = all([
                all(run_info["verification"]["data_verification"].values()),
                all(run_info["verification"]["model_verification"].values()),
                all(run_info["verification"]["training_verification"].values()),
                all(run_info["verification"]["privacy_verification"].values())
            ])
            run_info["verification"]["overall_status"] = "PASSED" if all_verified else "FAILED"

    return run_info

def generate_plots(run_id: str, run_info: Dict[str, Any], output_dir: Path):
    """Generate performance and privacy plots."""
    # Performance plots
    if run_info["training"]["epochs"] > 0:
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(run_info["training"]["train_loss"], label="Training Loss")
        plt.plot(run_info["training"]["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(run_info["training"]["train_acc"], label="Training Accuracy")
        plt.plot(run_info["training"]["val_acc"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{run_id}_performance.png", dpi=100)
        plt.close()
    
    # Privacy budget plot
    if run_info["privacy"]["privacy_budget"]:
        plt.figure(figsize=(8, 5))
        plt.plot(run_info["privacy"]["privacy_budget"], label="Privacy Budget")
        plt.axhline(y=run_info["privacy"]["target_epsilon"], color='r', linestyle='--', label="Target Epsilon")
        plt.xlabel("Epoch")
        plt.ylabel("Epsilon")
        plt.title("Privacy Budget Consumption")
        plt.legend()
        plt.savefig(output_dir / f"{run_id}_privacy.png", dpi=100)
        plt.close()

def generate_markdown_report(runs_info: dict, output_dir: Path):
    """Generate Markdown report from runs information."""
    # Ensure all required fields are present with defaults
    required_data_fields = {
        'dataset': 'N/A',
        'train_samples': 0,
        'test_samples': 0,
        'train_mean': 0.0,
        'train_std': 0.0,
        'test_mean': 0.0,
        'test_std': 0.0
    }
    required_model_fields = {
        'architecture': 'N/A',
        'total_parameters': 0,
        'num_layers': 0,
        'layers': []
    }
    for run in runs_info.values():
        for k, v in required_data_fields.items():
            if k not in run['data']:
                run['data'][k] = v
        for k, v in required_model_fields.items():
            if k not in run['model']:
                run['model'][k] = v

    markdown = f"""# MNIST Provenance Summary Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

This report summarizes the training runs and their outcomes for the MNIST dataset with differential privacy guarantees.

"""

    for run_id, run in runs_info.items():
        markdown += f"""
## Run: {run_id}

### Data Information
- Dataset: {run['data']['dataset']}
- Training Samples: {run['data']['train_samples']}
- Test Samples: {run['data']['test_samples']}
- Training Statistics:
  - Mean: {run['data']['train_mean']:.4f}
  - Std: {run['data']['train_std']:.4f}
- Test Statistics:
  - Mean: {run['data']['test_mean']:.4f}
  - Std: {run['data']['test_std']:.4f}

### Model Architecture
- Architecture: {run['model']['architecture']}
- Total Parameters: {run['model']['total_parameters']}
- Number of Layers: {run['model']['num_layers']}
- Layer Details:
"""
        for layer in run['model']['layers']:
            markdown += f"  - {layer['name']} ({layer['type']})\n"

        markdown += f"""
### Training Results
- Final Training Accuracy: {run['training']['final_train_acc']*100:.2f}%
- Final Validation Accuracy: {run['training']['final_val_acc']*100:.2f}%
- Number of Epochs: {run['training']['epochs']}
- Training Time: {run['training']['training_time']:.2f} seconds

![Performance Plots]({run_id}_performance.png)

### Privacy Guarantees
- Target Epsilon: {run['privacy']['target_epsilon']:.2f}
- Final Epsilon: {run['privacy']['final_epsilon']:.2f}
- Delta: {run['privacy']['delta']:.2e}

![Privacy Budget]({run_id}_privacy.png)

### Verification Results
Overall Status: {'✅ PASSED' if run['verification']['overall_status'] == 'PASSED' else '❌ FAILED'}

"""
        # Add verification details
        for key, value in run['verification'].items():
            if key != "overall_status":
                markdown += f"\n#### {key.replace('_', ' ').title()}\n"
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, bool):
                            status = "✅ Pass" if subvalue else "❌ Fail"
                            markdown += f"- {subkey.replace('_', ' ').title()}: {status}\n"
                        else:
                            markdown += f"- {subkey.replace('_', ' ').title()}: {subvalue}\n"
                else:
                    markdown += f"- {value}\n"

        markdown += "\n---\n"

    # Write the markdown file
    with open(output_dir / "summary_report.md", "w") as f:
        f.write(markdown)

def main():
    """Main function to generate the summary report."""
    # Create reports directory if it doesn't exist
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    # Get all provenance directories
    run_ids = get_provenance_dirs()
    if not run_ids:
        logger.error("No provenance directories found")
        return
    
    # Extract information from each run
    runs_info = {}
    for run_id in run_ids:
        logger.info(f"Processing run: {run_id}")
        run_info = extract_run_info(run_id)
        runs_info[run_id] = run_info
        
        # Generate plots for this run
        generate_plots(run_id, run_info, output_dir)
    
    # Generate Markdown report
    generate_markdown_report(runs_info, output_dir)
    logger.info(f"Summary report generated successfully at: {output_dir}/summary_report.md")

if __name__ == "__main__":
    main() 