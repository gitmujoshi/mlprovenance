"""
Business report generation for MNIST Provenance project.
This module provides functionality to generate comprehensive business reports
that summarize model performance, privacy guarantees, and verification status.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

class BusinessReport:
    """Generates comprehensive business reports for model training runs."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the business report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_training_metrics(self, metrics_file: str) -> Dict[str, Any]:
        """Load training metrics from JSON file."""
        with open(metrics_file, 'r') as f:
            return json.load(f)
            
    def _load_privacy_metrics(self, privacy_file: str) -> Dict[str, Any]:
        """Load privacy metrics from JSON file."""
        with open(privacy_file, 'r') as f:
            return json.load(f)
            
    def _load_verification_results(self, verification_file: str) -> Dict[str, Any]:
        """Load verification results from JSON file."""
        with open(verification_file, 'r') as f:
            return json.load(f)
            
    def _generate_performance_plots(self, metrics: Dict[str, Any], output_path: Path):
        """Generate performance visualization plots."""
        # Create performance plots
        plt.figure(figsize=(12, 6))
        
        # Plot training metrics
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics['train_acc'], label='Training Accuracy')
        plt.plot(metrics['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_plots.png')
        plt.close()
        
    def _generate_privacy_plots(self, privacy_metrics: Dict[str, Any], output_path: Path):
        """Generate privacy metrics visualization."""
        plt.figure(figsize=(10, 6))
        
        # Plot privacy budget consumption
        plt.plot(privacy_metrics['privacy_budget'], label='Privacy Budget')
        plt.axhline(y=privacy_metrics['target_epsilon'], color='r', linestyle='--', 
                   label='Target Epsilon')
        plt.title('Privacy Budget Consumption')
        plt.xlabel('Training Step')
        plt.ylabel('Epsilon')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'privacy_plots.png')
        plt.close()
        
    def _generate_html_report(self, 
                            metrics: Dict[str, Any],
                            privacy_metrics: Dict[str, Any],
                            verification_results: Dict[str, Any],
                            output_path: Path):
        """Generate HTML report using Jinja2 template."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MNIST Provenance Business Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 30px; }
                .metric { margin: 10px 0; }
                .status { padding: 5px 10px; border-radius: 3px; }
                .success { background-color: #d4edda; color: #155724; }
                .warning { background-color: #fff3cd; color: #856404; }
                .error { background-color: #f8d7da; color: #721c24; }
                img { max-width: 100%; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>MNIST Provenance Business Report</h1>
            <p>Generated on: {{ timestamp }}</p>
            
            <div class="section">
                <h2>Model Performance Summary</h2>
                <div class="metric">
                    <strong>Final Training Accuracy:</strong> {{ "%.2f"|format(metrics.final_train_acc * 100) }}%
                </div>
                <div class="metric">
                    <strong>Final Validation Accuracy:</strong> {{ "%.2f"|format(metrics.final_val_acc * 100) }}%
                </div>
                <div class="metric">
                    <strong>Training Time:</strong> {{ "%.2f"|format(metrics.training_time) }} seconds
                </div>
                <img src="performance_plots.png" alt="Performance Plots">
            </div>
            
            <div class="section">
                <h2>Privacy Guarantees</h2>
                <div class="metric">
                    <strong>Target Epsilon:</strong> {{ "%.2f"|format(privacy_metrics.target_epsilon) }}
                </div>
                <div class="metric">
                    <strong>Final Epsilon:</strong> {{ "%.2f"|format(privacy_metrics.final_epsilon) }}
                </div>
                <div class="metric">
                    <strong>Delta:</strong> {{ "%.2e"|format(privacy_metrics.delta) }}
                </div>
                <img src="privacy_plots.png" alt="Privacy Plots">
            </div>
            
            <div class="section">
                <h2>Verification Status</h2>
                {% for check, result in verification_results.items() %}
                <div class="metric">
                    <strong>{{ check }}:</strong>
                    <span class="status {{ result.status }}">
                        {{ result.status|upper }}
                    </span>
                    {% if result.message %}
                    <p>{{ result.message }}</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Generate recommendations based on metrics
        recommendations = []
        
        # Performance recommendations
        if metrics['final_val_acc'] < 0.95:
            recommendations.append(
                "Consider increasing model capacity or training time to improve accuracy"
            )
        if metrics['final_val_acc'] - metrics['final_train_acc'] > 0.1:
            recommendations.append(
                "Model shows signs of overfitting. Consider adding regularization"
            )
            
        # Privacy recommendations
        if privacy_metrics['final_epsilon'] > privacy_metrics['target_epsilon']:
            recommendations.append(
                "Privacy budget exceeded target. Consider reducing training steps or increasing noise"
            )
            
        # Verification recommendations
        if not all(r['status'] == 'success' for r in verification_results.values()):
            recommendations.append(
                "Some verification checks failed. Review model training process"
            )
            
        template = Template(template_str)
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics,
            privacy_metrics=privacy_metrics,
            verification_results=verification_results,
            recommendations=recommendations
        )
        
        with open(output_path / 'report.html', 'w') as f:
            f.write(html_content)
            
    def generate_report(self,
                       metrics_file: str,
                       privacy_file: str,
                       verification_file: str,
                       run_id: Optional[str] = None) -> str:
        """Generate a comprehensive business report.
        
        Args:
            metrics_file: Path to training metrics JSON file
            privacy_file: Path to privacy metrics JSON file
            verification_file: Path to verification results JSON file
            run_id: Optional identifier for this training run
            
        Returns:
            Path to the generated report
        """
        # Load all metrics
        metrics = self._load_training_metrics(metrics_file)
        privacy_metrics = self._load_privacy_metrics(privacy_file)
        verification_results = self._load_verification_results(verification_file)
        
        # Create run-specific directory
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        self._generate_performance_plots(metrics, run_dir)
        self._generate_privacy_plots(privacy_metrics, run_dir)
        
        # Generate HTML report
        self._generate_html_report(metrics, privacy_metrics, verification_results, run_dir)
        
        return str(run_dir / 'report.html') 