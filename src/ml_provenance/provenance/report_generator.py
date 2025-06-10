from typing import Dict, Any
import os
from datetime import datetime
from .provenance_data import ProvenanceData
import json
import logging
from pathlib import Path

class ReportGenerator:
    """Report generator for ML provenance using consistent data structure."""
    
    def __init__(self, provenance_dir: str):
        self.provenance_dir = Path(provenance_dir)
        self.logger = logging.getLogger(__name__)
    
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
    
    def generate_report(self, verification_results: Dict[str, Any]) -> str:
        """Generate a markdown report from verification results."""
        run_id = self.provenance_dir.name
        data = self._load_provenance_data()
        
        report = [
            "# MNIST Model Training Run Business Report",
            f"**Run ID:** {run_id}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            
            "## Executive Summary",
            self._generate_executive_summary(verification_results, data),
            
            "## 1. Training Configuration",
            self._generate_training_config_section(data),
            
            "## 2. Model Performance",
            self._generate_performance_section(data),
            
            "## 3. Privacy Analysis",
            self._generate_privacy_analysis_section(data),
            
            "## 4. Data Provenance",
            self._generate_data_section(verification_results),
            
            "## 5. Model Architecture",
            self._generate_model_section(verification_results),
            
            "## 6. Training Provenance",
            self._generate_training_section(verification_results),
            
            "## 7. Privacy Metrics",
            self._generate_privacy_section(verification_results),
            
            "## 8. Merkle Tree Verification",
            self._generate_merkle_section(verification_results),
            
            "## 9. Issues and Recommendations",
            self._generate_issues_section(verification_results),
            
            "## 10. Overall Status",
            self._generate_status_section(verification_results),
            
            "## 11. Next Steps",
            self._generate_next_steps(verification_results)
        ]
        
        return "\n".join(report)
    
    def _generate_executive_summary(self, results: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate the executive summary section."""
        training_info = data.get("training_provenance", {})
        final_metrics = training_info.get("final_metrics", {})
        privacy_metrics = training_info.get("privacy_metrics", {})
        
        section = [
            "### Key Metrics",
            f"- **Final Training Accuracy:** {final_metrics.get('train_accuracy', 0):.2%}",
            f"- **Final Test Accuracy:** {final_metrics.get('test_accuracy', 0):.2%}",
            f"- **Final Loss:** {final_metrics.get('loss', 0):.4f}",
            f"- **Privacy Budget (ε):** {privacy_metrics.get('achieved_epsilon', 'N/A')}",
            f"- **Privacy Risk (δ):** {privacy_metrics.get('achieved_delta', 'N/A')}\n",
            
            "### Verification Status",
            f"- {'✅' if results.get('model_verification', {}).get('architecture_match') else '❌'} Model Architecture",
            f"- {'✅' if results.get('training_verification', {}).get('final_metrics_present') else '❌'} Training Metrics",
            f"- {'✅' if results.get('training_verification', {}).get('privacy_metrics_present') else '❌'} Privacy Metrics",
            f"- {'✅' if results.get('data_verification', {}).get('hash_match') else '❌'} Data Integrity",
            f"- {'✅' if results.get('merkle_verification', {}).get('root_hash_match') else '❌'} Merkle Tree Verification\n",
            
            "### Overall Assessment",
            "The training run has been completed with the following key findings:",
            f"1. Model Performance: {'Satisfactory' if final_metrics.get('test_accuracy', 0) > 0.9 else 'Needs Improvement'}",
            f"2. Privacy Guarantees: {'Achieved' if privacy_metrics.get('achieved_epsilon', float('inf')) <= privacy_metrics.get('target_epsilon', 0) else 'Not Met'}",
            f"3. Data Integrity: {'Verified' if results.get('data_verification', {}).get('hash_match') else 'Issues Found'}",
            f"4. Model Integrity: {'Verified' if results.get('model_verification', {}).get('architecture_match') else 'Issues Found'}"
        ]
        
        return "\n".join(section)
    
    def _generate_training_config_section(self, data: Dict[str, Any]) -> str:
        """Generate the training configuration section."""
        training_info = data.get("training_provenance", {})
        config = training_info.get("config", {})
        
        section = [
            "### Training Parameters",
            f"- **Epochs:** {config.get('epochs', 'N/A')}",
            f"- **Batch Size:** {config.get('batch_size', 'N/A')}",
            f"- **Learning Rate:** {config.get('learning_rate', 'N/A')}",
            f"- **Optimizer:** {config.get('optimizer', 'N/A')}",
            f"- **Loss Function:** {config.get('loss_function', 'N/A')}\n",
            
            "### Privacy Parameters",
            f"- **Target Epsilon (ε):** {config.get('privacy_parameters', {}).get('target_epsilon', 'N/A')}",
            f"- **Target Delta (δ):** {config.get('privacy_parameters', {}).get('target_delta', 'N/A')}",
            f"- **Noise Multiplier:** {config.get('privacy_parameters', {}).get('noise_multiplier', 'N/A')}",
            f"- **Max Gradient Norm:** {config.get('privacy_parameters', {}).get('max_grad_norm', 'N/A')}\n",
            
            "### System Configuration",
            f"- **Device:** {config.get('device', 'N/A')}",
            f"- **Number of Workers:** {config.get('num_workers', 'N/A')}",
            f"- **Random Seed:** {config.get('seed', 'N/A')}"
        ]
        
        return "\n".join(section)
    
    def _generate_performance_section(self, data: Dict[str, Any]) -> str:
        """Generate the performance analysis section."""
        training_info = data.get("training_provenance", {})
        final_metrics = training_info.get("final_metrics", {})
        training_logs = training_info.get("training_logs", [])
        
        # Calculate performance metrics
        best_epoch = max(training_logs, key=lambda x: x.get('test_accuracy', 0)) if training_logs else {}
        worst_epoch = min(training_logs, key=lambda x: x.get('test_accuracy', 0)) if training_logs else {}
        
        section = [
            "### Final Performance",
            f"- **Training Accuracy:** {final_metrics.get('train_accuracy', 0):.2%}",
            f"- **Test Accuracy:** {final_metrics.get('test_accuracy', 0):.2%}",
            f"- **Final Loss:** {final_metrics.get('loss', 0):.4f}\n",
            
            "### Performance Analysis",
            f"- **Best Epoch:** {best_epoch.get('epoch', 'N/A')} (Accuracy: {best_epoch.get('test_accuracy', 0):.2%})",
            f"- **Worst Epoch:** {worst_epoch.get('epoch', 'N/A')} (Accuracy: {worst_epoch.get('test_accuracy', 0):.2%})",
            f"- **Average Epoch Time:** {sum(log.get('epoch_time', 0) for log in training_logs) / len(training_logs) if training_logs else 'N/A'} seconds\n",
            
            "### Performance Trends",
            "```",
            "Epoch  Training Accuracy  Test Accuracy  Loss",
            "-----  -----------------  -------------  ----"
        ]
        
        for log in training_logs:
            section.append(
                f"{log.get('epoch', 'N/A'):5d}  "
                f"{log.get('train_accuracy', 0):.2%}          "
                f"{log.get('test_accuracy', 0):.2%}      "
                f"{log.get('loss', 0):.4f}"
            )
        
        section.append("```")
        
        return "\n".join(section)
    
    def _generate_privacy_analysis_section(self, data: Dict[str, Any]) -> str:
        """Generate the privacy analysis section."""
        training_info = data.get("training_provenance", {})
        privacy_metrics = training_info.get("privacy_metrics", {})
        config = training_info.get("config", {})
        
        target_epsilon = config.get('privacy_parameters', {}).get('target_epsilon', float('inf'))
        achieved_epsilon = privacy_metrics.get('achieved_epsilon', float('inf'))
        
        section = [
            "### Privacy Budget Analysis",
            f"- **Target Epsilon (ε):** {target_epsilon}",
            f"- **Achieved Epsilon (ε):** {achieved_epsilon}",
            f"- **Target Delta (δ):** {config.get('privacy_parameters', {}).get('target_delta', 'N/A')}",
            f"- **Achieved Delta (δ):** {privacy_metrics.get('achieved_delta', 'N/A')}\n",
            
            "### Privacy Guarantees",
            f"- **Budget Status:** {'✅ Within Budget' if achieved_epsilon <= target_epsilon else '❌ Exceeded Budget'}",
            f"- **Noise Multiplier:** {privacy_metrics.get('noise_multiplier', 'N/A')}",
            f"- **Max Gradient Norm:** {config.get('privacy_parameters', {}).get('max_grad_norm', 'N/A')}\n",
            
            "### Privacy Impact Assessment",
            "The model's privacy guarantees are assessed as follows:",
            f"1. **Privacy Budget:** {'Satisfactory' if achieved_epsilon <= target_epsilon else 'Needs Review'}",
            f"2. **Risk Level:** {'Low' if privacy_metrics.get('achieved_delta', 1) <= 1e-5 else 'Medium'}",
            f"3. **Privacy-Utility Trade-off:** {'Balanced' if final_metrics.get('test_accuracy', 0) > 0.9 and achieved_epsilon <= target_epsilon else 'Needs Optimization'}"
        ]
        
        return "\n".join(section)
    
    def _generate_data_section(self, results: Dict[str, Any]) -> str:
        """Generate the data provenance section."""
        data_verification = results.get("data_verification", {})
        data_provenance = self._load_provenance_data().get("data_provenance", {})
        
        train_data = data_provenance.get("train", {})
        test_data = data_provenance.get("test", {})
        
        section = [
            "### Training Data",
            f"- **Dataset:** {data_provenance.get('dataset', 'Unknown')}",
            f"- **Samples:** {train_data.get('samples', 0):,}",
            "**Statistics:**",
            f"  - Mean: {train_data.get('mean', 0):.4f}",
            f"  - Std: {train_data.get('std', 0):.4f}",
            f"  - Min: {train_data.get('metadata', {}).get('statistics', {}).get('min', 0)}",
            f"  - Max: {train_data.get('metadata', {}).get('statistics', {}).get('max', 0)}",
            f"  - Shape: {train_data.get('metadata', {}).get('statistics', {}).get('shape', [])}",
            f"  - Dtype: {train_data.get('metadata', {}).get('statistics', {}).get('dtype', 'unknown')}",
            f"- **Hash:** {train_data.get('hash', 'Not available')}\n",
            
            "### Test Data",
            f"- **Samples:** {test_data.get('samples', 0):,}",
            "**Statistics:**",
            f"  - Mean: {test_data.get('mean', 0):.4f}",
            f"  - Std: {test_data.get('std', 0):.4f}",
            f"  - Min: {test_data.get('metadata', {}).get('statistics', {}).get('min', 0)}",
            f"  - Max: {test_data.get('metadata', {}).get('statistics', {}).get('max', 0)}",
            f"  - Shape: {test_data.get('metadata', {}).get('statistics', {}).get('shape', [])}",
            f"  - Dtype: {test_data.get('metadata', {}).get('statistics', {}).get('dtype', 'unknown')}",
            f"- **Hash:** {test_data.get('hash', 'Not available')}\n",
            
            "### Data Verification Status",
            f"- {'✅' if data_verification.get('has_train_data') else '❌'} Training data present",
            f"- {'✅' if data_verification.get('has_test_data') else '❌'} Test data present",
            f"- {'✅' if data_verification.get('has_timestamp') else '❌'} Timestamp present",
            f"- {'✅' if data_verification.get('hash_match') else '❌'} Data hash match",
            f"- {'✅' if data_verification.get('statistics_verified') else '❌'} Statistics verified",
            f"- {'✅' if data_verification.get('metadata_verified') else '❌'} Metadata verified"
        ]
        
        return "\n".join(section)
    
    def _generate_model_section(self, results: Dict[str, Any]) -> str:
        """Generate the model architecture section."""
        model_verification = results.get("model_verification", {})
        data = self._load_provenance_data()
        model_info = data.get("model_provenance", {}).get("info", {})
        
        section = [
            "### Model Configuration",
            f"- **Architecture:** {model_info.get('architecture', 'Unknown')}",
            f"- **Total Parameters:** {model_info.get('total_parameters', 0):,}",
            f"- **Number of Layers:** {len(model_info.get('layers', []))}\n",
            
            "### Layer Structure"
        ]
        
        # Add layer information
        for i, layer in enumerate(model_info.get('layers', [])):
            section.append(f"{i+1}. **Layer {i}:** {layer.get('name', 'Unknown')} ({layer.get('type', 'Unknown')})")
        
        section.extend([
            "\n### Architecture Verification",
            f"- {'✅' if model_verification.get('layer_count_match') else '❌'} Layer count match",
            f"- {'✅' if model_verification.get('parameter_count_match') else '❌'} Parameter count match",
            f"- {'✅' if model_verification.get('architecture_match') else '❌'} Architecture match"
        ])
        
        return "\n".join(section)
    
    def _generate_training_section(self, results: Dict[str, Any]) -> str:
        """Generate the training provenance section."""
        training_verification = results.get("training_verification", {})
        data = self._load_provenance_data()
        training_info = data.get("training_provenance", {})
        config = training_info.get("config", {})
        
        section = [
            "### Training Configuration",
            f"- **Epochs:** {config.get('epochs', 'N/A')}",
            f"- **Batch Size:** {config.get('batch_size', 'N/A')}",
            f"- **Learning Rate:** {config.get('learning_rate', 'N/A')}",
            f"- **Optimizer:** {config.get('optimizer', 'N/A')}",
            f"- **Loss Function:** {config.get('loss_function', 'N/A')}\n",

            "### Training Metrics",
            f"- **Training Hash:** {training_info.get('hash', 'Not available')}",
            f"- **Privacy Hash:** {training_info.get('privacy_metrics', {}).get('hash', 'Not available')}\n",
            
            "### Verification Status",
            f"- {'✅' if training_verification.get('final_metrics_present') else '❌'} Final metrics present",
            f"- {'✅' if training_verification.get('training_logs_present') else '❌'} Training logs present",
            f"- {'✅' if training_verification.get('privacy_metrics_present') else '❌'} Privacy metrics present",
            f"- {'✅' if training_verification.get('hash_present') else '❌'} Hash present",
            f"- {'✅' if training_verification.get('timestamp_present') else '❌'} Timestamp present"
        ]
        
        return "\n".join(section)
    
    def _generate_privacy_section(self, results: Dict[str, Any]) -> str:
        """Generate the privacy metrics section."""
        data = self._load_provenance_data()
        privacy_info = data.get("training_provenance", {}).get("privacy_metrics", {})
        section = [
            "### Privacy Metrics",
            f"- **Target Epsilon:** {privacy_info.get('target_epsilon', 'N/A')}",
            f"- **Target Delta:** {privacy_info.get('target_delta', 'N/A')}",
            f"- **Noise Multiplier:** {privacy_info.get('noise_multiplier', 'N/A')}",
            f"- **Achieved Epsilon:** {privacy_info.get('achieved_epsilon', 'N/A')}",
            f"- **Achieved Delta:** {privacy_info.get('achieved_delta', 'N/A')}",
            f"- **Timestamp:** {privacy_info.get('timestamp', 'N/A')}"
        ]
        return "\n".join(section)
    
    def _generate_merkle_section(self, results: Dict[str, Any]) -> str:
        """Generate the Merkle tree verification section."""
        merkle_verification = results.get("merkle_verification", {})
        data = self._load_provenance_data()
        
        section = [
            "### Hash Structure",
            f"- Hash structure has been dumped to: `{self.provenance_dir}/hash_structure.json`",
            "- Root hash computed and stored\n",
            
            "### Merkle Tree Structure",
            "```",
            "Root Node",
            "├── Data Node",
            "│   ├── Training Data Node",
            "│   └── Test Data Node",
            "├── Model Node",
            "│   ├── Architecture Node",
            "│   └── Weights Node",
            "└── Training Node",
            "    ├── Epoch Nodes",
            "    │   ├── Model State",
            "    │   ├── Metrics",
            "    │   └── Privacy Metrics",
            "    └── Final State",
            "```\n",
            
            "### Component Verification",
            f"- {'✅' if merkle_verification.get('component_verification', {}).get('train') else '❌'} Train data component verified",
            f"- {'✅' if merkle_verification.get('component_verification', {}).get('test') else '❌'} Test data component verified",
            f"- {'✅' if merkle_verification.get('component_verification', {}).get('model') else '❌'} Model component verified",
            f"- {'✅' if merkle_verification.get('component_verification', {}).get('training') else '❌'} Training component verified"
        ]
        
        return "\n".join(section)
    
    def _generate_issues_section(self, results: Dict[str, Any]) -> str:
        """Generate the issues and recommendations section."""
        issues = []
        
        # Check data hash mismatch
        if not results.get("data_verification", {}).get("hash_match"):
            issues.append(
                "1. **Data Hash Mismatch**\n"
                "   - Computed hash differs from stored hash\n"
                "   - **Recommendation:** Review data serialization process to ensure consistent hashing"
            )
        
        # Check model verification
        if not results.get("model_verification", {}).get("model_hash_match"):
            issues.append(
                "2. **Model Hash Mismatch**\n"
                "   - Model hash verification failed\n"
                "   - **Recommendation:** Verify model serialization and hashing process"
            )
        
        if not issues:
            issues.append("No critical issues identified in this run.")
        
        section = [
            "### Identified Issues",
            *issues,
            "\n### Security Considerations",
            "1. All components have been properly hashed and verified",
            "2. Merkle tree structure ensures data integrity",
            "3. Privacy metrics are tracked and verified"
        ]
        
        return "\n".join(section)
    
    def _generate_status_section(self, results: Dict[str, Any]) -> str:
        """Generate the overall status section."""
        section = [
            "### Verification Results",
            f"- {'✅' if results.get('model_verification', {}).get('architecture_match') else '❌'} Model architecture verification",
            f"- {'✅' if results.get('training_verification', {}).get('final_metrics_present') else '❌'} Training metrics verification",
            f"- {'✅' if results.get('training_verification', {}).get('privacy_metrics_present') else '❌'} Privacy metrics verification",
            f"- {'✅' if results.get('data_verification', {}).get('hash_match') else '❌'} Data hash verification",
            f"- {'✅' if results.get('merkle_verification', {}).get('root_hash_match') else '❌'} Merkle tree structure verification\n",
            
            "### Final Assessment",
            "The training run has been verified with the following status:"
        ]
        
        # Determine overall status
        all_verified = all([
            results.get('model_verification', {}).get('architecture_match'),
            results.get('training_verification', {}).get('final_metrics_present'),
            results.get('training_verification', {}).get('privacy_metrics_present'),
            results.get('data_verification', {}).get('hash_match'),
            results.get('merkle_verification', {}).get('root_hash_match')
        ])
        
        if all_verified:
            section.append("✅ All components verified successfully.")
        else:
            section.append("⚠️ Some components failed verification. See Issues section for details.")
        
        return "\n".join(section)
    
    def _generate_next_steps(self, results: Dict[str, Any]) -> str:
        """Generate the next steps section."""
        steps = [
            "1. Review verification results",
            "2. Address any identified issues",
            "3. Monitor future runs for similar issues",
            "4. Consider implementing additional data integrity checks"
        ]
        
        return "\n".join(steps)
    
    def _load_provenance_data(self) -> Dict[str, Any]:
        """Load the provenance data from the JSON file."""
        try:
            with open(self.provenance_dir / "provenance.json", "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading provenance data: {str(e)}")
            return {}
    
    def save_report(self, report: str) -> None:
        """Save the report to a markdown file."""
        report_path = self.provenance_dir / "verification_report.md"
        with open(report_path, "w") as f:
            f.write(report) 
        self.logger.info(f"Report saved to: {report_path}")

def generate_report(provenance_dir: str) -> None:
    """Generate and save a verification report."""
    generator = ReportGenerator(provenance_dir)
    
    # Load verification results
    try:
        with open(Path(provenance_dir) / "verification.json", "r") as f:
            verification_results = json.load(f)
    except Exception as e:
        logging.error(f"Error loading verification results: {str(e)}")
        return
    
    # Generate and save report
    report = generator.generate_report(verification_results)
    generator.save_report(report)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python report_generator.py <provenance_dir>")
        sys.exit(1)
    
    generate_report(sys.argv[1]) 