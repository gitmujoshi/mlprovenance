import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .safety_checks import SafetyConfig, SafetyChecker, ModelSafetyWrapper, AgeRating

logger = logging.getLogger(__name__)

class SafeTrainingPipeline:
    def __init__(
        self,
        model,
        safety_config: SafetyConfig,
        provenance_dir: str = "artifacts/provenance"
    ):
        self.model = model
        self.safety_checker = SafetyChecker(safety_config)
        self.safe_model = ModelSafetyWrapper(model, self.safety_checker)
        
        # Set up provenance tracking
        self.provenance_dir = Path(provenance_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.provenance_dir = self.provenance_dir / self.run_id
        self.provenance_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, train_data, **kwargs):
        """
        Train the model with safety checks and provenance tracking.
        """
        # Initialize training provenance
        training_provenance = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "training_params": kwargs,
            "safety_config": self.safety_checker.get_safety_provenance()
        }
        
        try:
            # Training loop
            for epoch in range(kwargs.get('epochs', 1)):
                logger.info(f"Starting epoch {epoch + 1}")
                
                for batch in train_data:
                    # Process batch with safety checks
                    safe_outputs = []
                    for input_text in batch:
                        output, safety_info = self.safe_model.generate(input_text)
                        safe_outputs.append(output)
                        
                        # Update provenance with safety metrics
                        self._save_metric(
                            "safety_metrics",
                            self.safety_checker.metrics.to_dict()
                        )
                    
                    # Update model with safe outputs
                    self.model.update(safe_outputs)
                
                # Save epoch-level safety provenance
                epoch_provenance = {
                    "epoch": epoch + 1,
                    "safety_metrics": self.safety_checker.metrics.to_dict()
                }
                self._save_metric("epoch_safety", epoch_provenance)
            
            # Save final safety provenance
            final_provenance = {
                "final_safety_metrics": self.safety_checker.metrics.to_dict(),
                "training_completed": True
            }
            self._save_metric("final_safety", final_provenance)
            
            # Save safety-specific provenance
            safety_provenance_path = self.provenance_dir / "safety_provenance.json"
            self.safety_checker.save_safety_provenance(str(safety_provenance_path))
            
            logger.info(f"Training completed. Safety provenance saved to {safety_provenance_path}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            # Save error provenance
            error_provenance = {
                "error": str(e),
                "safety_metrics": self.safety_checker.metrics.to_dict()
            }
            self._save_metric("error", error_provenance)
            raise
    
    def _save_metric(self, name: str, data: Dict):
        """Save a metric to the provenance directory."""
        import json
        metric_file = self.provenance_dir / f"{name}.json"
        with open(metric_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_safety_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive safety report for the training run.
        """
        metrics = self.safety_checker.metrics.to_dict()
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "safety_config": self.safety_checker.get_safety_provenance(),
            "metrics": metrics,
            "summary": {
                "total_checks": metrics["total_checks"],
                "pass_rate": metrics["pass_rate"],
                "content_warnings": metrics["content_warnings"],
                "most_common_sensitive_topics": sorted(
                    metrics["sensitive_topics_detected"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "most_common_filter_violations": sorted(
                    metrics["filter_violations"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        } 