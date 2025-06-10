#!/usr/bin/env python3
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import re

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

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
    run_info = {}
    
    def extract_metrics(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Recursively extract metrics from nested dictionary."""
        metrics = {}
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                nested_metrics = extract_metrics(value, current_key)
                metrics.update(nested_metrics)
            elif isinstance(value, (int, float, str, bool)):
                # Store leaf values
                metrics[current_key] = value
            elif isinstance(value, list):
                # Handle lists (e.g., training history)
                if all(isinstance(item, dict) for item in value):
                    # If list of dictionaries, process each item
                    for i, item in enumerate(value):
                        nested_metrics = extract_metrics(item, f"{current_key}[{i}]")
                        metrics.update(nested_metrics)
                else:
                    # Store list as is
                    metrics[current_key] = value
        return metrics

    def build_structure_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build a nested structure from flattened metrics."""
        structure = {}
        
        for key, value in metrics.items():
            # Split the key into components
            parts = key.split('.')
            current = structure
            
            # Build the nested structure
            for i, part in enumerate(parts[:-1]):
                # Handle array indices
                if '[' in part and ']' in part:
                    base_part, idx = part.split('[')
                    idx = int(idx.rstrip(']'))
                    
                    if base_part not in current:
                        current[base_part] = []
                    while len(current[base_part]) <= idx:
                        current[base_part].append({})
                    current = current[base_part][idx]
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            
            # Set the value
            last_part = parts[-1]
            if '[' in last_part and ']' in last_part:
                base_part, idx = last_part.split('[')
                idx = int(idx.rstrip(']'))
                
                if base_part not in current:
                    current[base_part] = []
                while len(current[base_part]) <= idx:
                    current[base_part].append(None)
                current[base_part][idx] = value
            else:
                current[last_part] = value
        
        return structure

    # Read provenance.json
    provenance_path = Path(f"artifacts/provenance/{run_id}/provenance.json")
    if provenance_path.exists():
        with open(provenance_path) as f:
            provenance_data = json.load(f)
            logger.info(f"Loaded provenance.json for run {run_id}")
            
            # Debug logging for raw data
            logger.info(f"\nRaw provenance data for run {run_id}:")
            logger.info(json.dumps(provenance_data, indent=2))
            
            # Extract all metrics recursively
            extracted_metrics = extract_metrics(provenance_data)
            
            # Debug logging for extracted metrics
            logger.info(f"\nExtracted metrics for run {run_id}:")
            logger.info(json.dumps(extracted_metrics, indent=2))
            
            # Build structure from metrics
            run_info = build_structure_from_metrics(extracted_metrics)
            
            # Debug logging for final structure
            logger.info(f"\nFinal structure for run {run_id}:")
            logger.info(json.dumps(run_info, indent=2))
    else:
        logger.warning(f"provenance.json not found for run {run_id}")

    # Read verification.json
    verification_path = Path(f"artifacts/provenance/{run_id}/verification.json")
    if verification_path.exists():
        with open(verification_path) as f:
            verification_data = json.load(f)
            logger.info(f"Loaded verification.json for run {run_id}")
            run_info["verification"] = verification_data
    else:
        logger.warning(f"verification.json not found for run {run_id}")

    return run_info

def generate_plots(run_id: str, run_info: Dict[str, Any], output_dir: Path) -> None:
    """Generate plots for a run."""
    # Set up the plot style
    plt.style.use('ggplot')
    
    # Create plots directory if it doesn't exist
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    def find_metric_arrays(data: Dict[str, Any], prefix: str = "") -> Dict[str, List[float]]:
        """Recursively find all numeric arrays in the data structure."""
        arrays = {}
        
        def is_numeric_array(value: Any) -> bool:
            """Check if a value is a numeric array."""
            if not isinstance(value, list):
                return False
            return all(isinstance(x, (int, float)) for x in value)
        
        def process_value(key: str, value: Any):
            current_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                for k, v in value.items():
                    process_value(k, v)
            elif is_numeric_array(value):
                # Found a numeric array
                arrays[current_key] = value
            elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
                # Process list of dictionaries
                for i, item in enumerate(value):
                    for k, v in item.items():
                        process_value(f"{k}[{i}]", v)
        
        # Process all items in the data
        for key, value in data.items():
            process_value(key, value)
        
        return arrays
    
    def generate_plot(data: List[float], title: str, xlabel: str, ylabel: str, 
                     filename: str, color: str = 'b-', label: str = None) -> None:
        """Generate a single plot."""
        if not data or len(data) == 0:
            logger.warning(f"No data to plot for {title}")
            return
            
        plt.figure(figsize=(10, 6))
        x_values = list(range(1, len(data) + 1))
        plt.plot(x_values, data, color, label=label or title, linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if label:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Generated plot {filename} with {len(data)} points")
    
    # Find all numeric arrays in the data
    metric_arrays = find_metric_arrays(run_info)
    
    # Debug logging
    logger.info(f"\nFound metric arrays for {run_id}:")
    for key, values in metric_arrays.items():
        logger.info(f"{key}: {len(values)} points")
    
    # Track which plots we've generated
    generated_plots = set()
    
    # Generate plots for each array
    for key, values in metric_arrays.items():
        # Determine plot parameters based on the key
        if 'loss' in key.lower():
            if 'train_loss' in key.lower() and 'train_loss' not in generated_plots:
                generate_plot(
                    values,
                    'Training Loss Over Time',
                    'Epoch',
                    'Loss',
                    f"{run_id}_loss.png",
                    'b-',
                    'Training Loss'
                )
                generated_plots.add('train_loss')
            elif 'val_loss' in key.lower() and 'val_loss' not in generated_plots:
                generate_plot(
                    values,
                    'Validation Loss Over Time',
                    'Epoch',
                    'Loss',
                    f"{run_id}_val_loss.png",
                    'r-',
                    'Validation Loss'
                )
                generated_plots.add('val_loss')
        elif 'acc' in key.lower() or 'accuracy' in key.lower():
            if 'train_acc' in key.lower() and 'train_acc' not in generated_plots:
                generate_plot(
                    values,
                    'Training Accuracy Over Time',
                    'Epoch',
                    'Accuracy',
                    f"{run_id}_train_acc.png",
                    'g-',
                    'Training Accuracy'
                )
                generated_plots.add('train_acc')
            elif 'val_acc' in key.lower() and 'val_acc' not in generated_plots:
                generate_plot(
                    values,
                    'Validation Accuracy Over Time',
                    'Epoch',
                    'Accuracy',
                    f"{run_id}_val_acc.png",
                    'r-',
                    'Validation Accuracy'
                )
                generated_plots.add('val_acc')
        elif 'privacy' in key.lower() or 'epsilon' in key.lower():
            if 'privacy_budget' not in generated_plots:
                generate_plot(
                    values,
                    'Privacy Budget Over Time',
                    'Step',
                    'Epsilon',
                    f"{run_id}_privacy.png",
                    'm-',
                    'Privacy Budget'
                )
                generated_plots.add('privacy_budget')
    
    logger.info(f"Plot generation completed for run {run_id}")

def generate_html_report(runs_info: dict, output_dir: Path):
    """Generate HTML report from runs information."""
    logger.info(f"Generating HTML report for {len(runs_info)} runs")
    for run_id, run_info in runs_info.items():
        logger.info(f"Run {run_id} data: {json.dumps(run_info, indent=2)}")

    def format_value(value):
        """Format different types of values for display."""
        if isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], (int, float)):
                return f"{value[-1]:.4f}"  # Show last value for numeric lists
            return str(value)
        return str(value)

    def generate_section(data, title=None):
        """Recursively generate HTML sections from nested dictionaries."""
        html = []
        if title:
            html.append(f"<h3>{title}</h3>")
        
        if isinstance(data, dict):
            # Check if this is a list-like dictionary (e.g., training history)
            if all(isinstance(k, int) for k in data.keys()):
                # This is likely a list-like structure, create a table
                html.append("<table class='epoch-table'>")
                # Get all possible columns from the first item
                if data:
                    first_item = next(iter(data.values()))
                    if isinstance(first_item, dict):
                        html.append("<tr>")
                        html.append("<th>Index</th>")
                        for key in first_item.keys():
                            html.append(f"<th>{key.replace('_', ' ').title()}</th>")
                        html.append("</tr>")
                        
                        for idx, item in data.items():
                            html.append("<tr>")
                            html.append(f"<td>{idx}</td>")
                            for value in item.values():
                                html.append(f"<td>{format_value(value)}</td>")
                            html.append("</tr>")
                html.append("</table>")
            else:
                # Regular dictionary, create a table
                html.append("<table>")
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        # For nested structures, create a new section
                        html.append(f"<tr><th>{key.replace('_', ' ').title()}</th><td>")
                        html.extend(generate_section(value))
                        html.append("</td></tr>")
                    else:
                        html.append(f"<tr><th>{key.replace('_', ' ').title()}</th><td>{format_value(value)}</td></tr>")
                html.append("</table>")
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                # List of dictionaries, create a table
                html.append("<table class='epoch-table'>")
                # Get all possible columns
                if data:
                    html.append("<tr>")
                    for key in data[0].keys():
                        html.append(f"<th>{key.replace('_', ' ').title()}</th>")
                    html.append("</tr>")
                    
                    for item in data:
                        html.append("<tr>")
                        for value in item.values():
                            html.append(f"<td>{format_value(value)}</td>")
                        html.append("</tr>")
                html.append("</table>")
            else:
                # Simple list, show as comma-separated values
                html.append(f"<p>{', '.join(format_value(v) for v in data)}</p>")
        
        return html

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unified Training Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .run-section { margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 15px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            h2 { color: #333; }
            h3 { color: #666; }
            .epoch-table { margin-top: 10px; }
            .plot-container { 
                display: flex; 
                flex-wrap: wrap; 
                gap: 20px; 
                margin: 20px 0; 
            }
            .plot { 
                flex: 1; 
                min-width: 300px; 
                text-align: center; 
            }
            .plot img { 
                max-width: 100%; 
                height: auto; 
            }
        </style>
    </head>
    <body>
        <h1>Unified Training Report</h1>
    """
    
    for run_id, run_info in runs_info.items():
        html_content += f"""
        <div class="run-section">
            <h2>Run: {run_id}</h2>
        """
        
        # Add plots if they exist
        performance_plot = output_dir / f"{run_id}_performance.png"
        privacy_plot = output_dir / f"{run_id}_privacy.png"
        
        if performance_plot.exists() or privacy_plot.exists():
            html_content += '<div class="plot-container">'
            if performance_plot.exists():
                html_content += f"""
                <div class="plot">
                    <h3>Performance Metrics</h3>
                    <img src="{performance_plot.name}" alt="Performance Plot">
                </div>
                """
            if privacy_plot.exists():
                html_content += f"""
                <div class="plot">
                    <h3>Privacy Budget</h3>
                    <img src="{privacy_plot.name}" alt="Privacy Plot">
                </div>
                """
            html_content += '</div>'
        
        html_content += "\n".join(generate_section(run_info))
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_dir / "unified_report.html", "w") as f:
        f.write(html_content)
    logger.info(f"HTML report generated successfully at: {output_dir}/unified_report.html")

def generate_detailed_report(run_id: str, run_info: Dict[str, Any], output_dir: Path) -> None:
    """Generate a detailed report for a run."""
    report_path = output_dir / f"{run_id}_detailed_report.md"
    
    def format_value(value: Any) -> str:
        """Format a value for display in the report."""
        if isinstance(value, (int, float)):
            return f"{value:.4f}" if isinstance(value, float) else str(value)
        elif isinstance(value, list):
            if len(value) == 0:
                return "[]"
            if all(isinstance(x, (int, float)) for x in value):
                return f"[{', '.join(format_value(x) for x in value[:5])}{'...' if len(value) > 5 else ''}]"
            return f"[{len(value)} items]"
        elif isinstance(value, dict):
            return f"{{ {len(value)} items }}"
        return str(value)
    
    def process_section(data: Dict[str, Any], section_name: str, level: int = 1) -> List[str]:
        """Process a section of the data recursively."""
        lines = []
        prefix = '#' * level
        
        # Add section header
        lines.append(f"\n{prefix} {section_name}\n")
        
        # Process each item in the section
        for key, value in data.items():
            if isinstance(value, dict):
                # For nested dictionaries, create a subsection
                lines.extend(process_section(value, key.replace('_', ' ').title(), level + 1))
            elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
                # For lists of dictionaries, create a subsection for each item
                lines.append(f"\n{prefix}# {key.replace('_', ' ').title()}\n")
                for i, item in enumerate(value):
                    lines.append(f"\n{prefix}## Item {i+1}\n")
                    lines.extend(process_section(item, "", level + 2))
            else:
                # For simple values, add them to a table
                if not lines or not lines[-1].startswith('|'):
                    lines.append("\n| Metric | Value |")
                    lines.append("|--------|-------|")
                lines.append(f"| {key.replace('_', ' ').title()} | {format_value(value)} |")
        
        return lines
    
    # Start with the run ID
    content = [f"# Detailed Report for Run {run_id}\n"]
    
    # Process the entire run_info structure
    content.extend(process_section(run_info, "Run Information"))
    
    # Write the report
    with open(report_path, 'w') as f:
        f.write('\n'.join(content))
    
    logger.info(f"Generated detailed report at {report_path}")

def generate_unified_report(provenance_dir: Path, output_dir: Path) -> None:
    """Generate a unified report from all provenance files."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all run directories
    run_dirs = [d for d in provenance_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        logger.warning(f"No run directories found in {provenance_dir}")
        return
    
    # Process each run
    for run_dir in run_dirs:
        run_id = run_dir.name
        logger.info(f"Processing run {run_id}")
        
        # Extract run information
        run_info = extract_run_info(run_id)
        if not run_info:
            logger.warning(f"No run information found for {run_id}")
            continue
        
        # Generate reports
        generate_detailed_report(run_id, run_info, run_dir)
        generate_plots(run_id, run_info, run_dir)
    
    logger.info(f"Unified report generation completed. Output directory: {output_dir}")

def main():
    """Main function to generate the unified report."""
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
        if run_info:
            runs_info[run_id] = run_info
            
            # Create run-specific output directory
            run_dir = Path(f"artifacts/provenance/{run_id}")
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate reports
            generate_detailed_report(run_id, run_info, run_dir)
            generate_plots(run_id, run_info, run_dir)
    
    # Generate HTML report in the latest run's directory
    if run_ids:
        latest_run_dir = Path(f"artifacts/provenance/{run_ids[0]}")
        generate_html_report(runs_info, latest_run_dir)
        logger.info(f"Unified report generation completed. Output directory: {latest_run_dir}")

if __name__ == "__main__":
    main() 