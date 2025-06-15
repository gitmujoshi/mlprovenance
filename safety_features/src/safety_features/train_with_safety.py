import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import torch
import platform
import sys

from .safety_checks import SafetyConfig, SafetyChecker, ModelSafetyWrapper, AgeRating

logger = logging.getLogger(__name__)

class SafeTrainingPipeline:
    def __init__(self, model, safety_config: SafetyConfig, provenance_dir: str):
        self.model = model
        self.safety_config = safety_config
        self.safety_checker = SafetyChecker(safety_config)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.provenance_dir = Path(provenance_dir) / self.timestamp
        self.provenance_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_file = self.provenance_dir / "safety_provenance.json"
        
        # Initialize system info
        self.system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device": str(self.model.device)
        }
        
        logger.info("SafeTrainingPipeline initialized with:")
        logger.info(f"  Safety config: {safety_config.to_dict()}")
        logger.info(f"  System info: {self.system_info}")
        
    def train(self, train_data, epochs: int, batch_size: int):
        """
        Train the model with safety checks.
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Batch size: {batch_size}")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_metrics = {
                "epoch": epoch + 1,
                "total_checks": 0,
                "passed_checks": 0,
                "content_warnings": 0,
                "sensitive_topics": {},
                "filter_violations": {},
                "losses": [],
                "batch_metrics": []
            }
            
            for batch_idx, batch in enumerate(tqdm(train_data, desc=f"Epoch {epoch + 1}")):
                try:
                    # Apply safety checks to batch
                    safe_texts = self.safety_checker.check_batch(batch)
                    if not safe_texts:
                        logger.warning(f"Batch {batch_idx}: No safe texts found, skipping")
                        continue
                        
                    # Update model with safe texts
                    loss = self.model.update(safe_texts)
                    if loss is not None:
                        epoch_metrics["losses"].append(loss)
                        
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        avg_loss = sum(epoch_metrics["losses"][-10:]) / min(10, len(epoch_metrics["losses"][-10:]))
                        logger.info(f"Batch {batch_idx + 1}: Average loss = {avg_loss:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
                    
            # Calculate epoch metrics
            epoch_avg_loss = sum(epoch_metrics["losses"]) / len(epoch_metrics["losses"]) if epoch_metrics["losses"] else 0
            logger.info(f"Epoch {epoch + 1} complete:")
            logger.info(f"  Average loss: {epoch_avg_loss:.4f}")
            logger.info(f"  Total batches processed: {len(epoch_metrics['losses'])}")
            
            # Save safety report after each epoch
            self._save_safety_report(epoch_metrics)
            
    def _save_safety_report(self, epoch_metrics: Dict[str, Any]):
        """
        Save safety report for the current epoch.
        """
        report = {
            "timestamp": self.timestamp,
            "safety_config": self.safety_config.to_dict(),
            "system_info": self.system_info,
            "epoch_metrics": epoch_metrics,
            "summary": {
                "total_checks": self.safety_checker.metrics.total_checks,
                "pass_rate": self.safety_checker.metrics.passed_checks / self.safety_checker.metrics.total_checks if self.safety_checker.metrics.total_checks > 0 else 0,
                "content_warnings": self.safety_checker.metrics.content_warnings,
                "most_common_sensitive_topics": sorted(
                    self.safety_checker.metrics.sensitive_topics_detected.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "most_common_filter_violations": sorted(
                    self.safety_checker.metrics.filter_violations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        }
        
        with open(self.provenance_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Safety report saved to {self.provenance_file}") 