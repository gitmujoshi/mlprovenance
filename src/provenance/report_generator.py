from typing import Dict, Any
import os
from datetime import datetime
from .provenance_data import ProvenanceData

class ReportGenerator:
    """Report generator for ML provenance using consistent data structure."""
    
    def __init__(self, provenance_data: ProvenanceData):
        self.provenance = provenance_data
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display."""
        if not metrics:
            return "No metrics available"
        return "\n".join(f"- {k}: {v:.4f}" for k, v in metrics.items())
    
    def _format_verification_results(self, results: Dict[str, bool]) -> str:
        """Format verification results for display."""
        if not results:
            return "No verification results available"
        return "\n".join(f"- {k}: {'✅' if v else '❌'}" for k, v in results.items())
    
    def generate_report(self, verification_results: Dict[str, Dict[str, bool]]) -> str:
        """Generate a comprehensive report."""
        report = [
            "# ML Provenance Report",
            f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Version: {self.provenance.data['version']}",
            
            "\n## Data Provenance",
            f"Dataset: {self.provenance.data['data']['metadata']['dataset']}",
            f"Train Samples: {self.provenance.data['data']['metadata']['train_samples']}",
            f"Test Samples: {self.provenance.data['data']['metadata']['test_samples']}",
            f"Timestamp: {self.provenance.data['data']['metadata']['timestamp']}",
            "\n### Data Verification",
            self._format_verification_results(verification_results.get('data', {})),
            
            "\n## Model Provenance",
            f"Architecture: {self.provenance.data['model']['metadata']['architecture']}",
            f"Parameters: {self.provenance.data['model']['metadata']['parameters']:,}",
            f"Timestamp: {self.provenance.data['model']['metadata']['timestamp']}",
            "\n### Model Verification",
            self._format_verification_results(verification_results.get('model', {})),
            
            "\n## Training Provenance",
            f"Epochs: {self.provenance.data['training']['metadata']['epochs']}",
            f"Batch Size: {self.provenance.data['training']['metadata']['batch_size']}",
            f"Learning Rate: {self.provenance.data['training']['metadata']['learning_rate']}",
            "\n### Final Metrics",
            self._format_metrics(self.provenance.data['training']['metadata']['final_metrics']),
            "\n### Training Verification",
            self._format_verification_results(verification_results.get('training', {})),
            
            "\n## Privacy Provenance",
            f"Target Epsilon: {self.provenance.data['privacy']['metadata']['target_epsilon']}",
            f"Target Delta: {self.provenance.data['privacy']['metadata']['target_delta']}",
            f"Noise Multiplier: {self.provenance.data['privacy']['metadata']['noise_multiplier']}",
            f"Achieved Epsilon: {self.provenance.data['privacy']['metadata']['achieved_epsilon']}",
            f"Achieved Delta: {self.provenance.data['privacy']['metadata']['achieved_delta']}",
            f"Timestamp: {self.provenance.data['privacy']['metadata']['timestamp']}",
            "\n### Privacy Verification",
            self._format_verification_results(verification_results.get('privacy', {}))
        ]
        
        return "\n".join(report)
    
    def save_report(self, filepath: str, verification_results: Dict[str, Dict[str, bool]]) -> None:
        """Save the report to a file."""
        report = self.generate_report(verification_results)
        with open(filepath, 'w') as f:
            f.write(report) 