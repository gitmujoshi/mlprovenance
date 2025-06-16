import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ..base import BaseSafetyTrainer
from .wrapper import PyTorchSafetyWrapper

class PyTorchSafetyTrainer(BaseSafetyTrainer):
    def __init__(self, model: nn.Module, config: Dict):
        super().__init__(model, config)
        self.safety_wrapper = PyTorchSafetyWrapper(config.get('safety_config'))
        
    def validate_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Validate training batch for safety"""
        if isinstance(batch, (tuple, list)):
            # Handle multiple inputs (e.g., input and target)
            return tuple(self.safety_wrapper.validate_input(x) for x in batch)
        return self.safety_wrapper.validate_input(batch)
        
    def compute_loss(self, output: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute training loss with safety penalty"""
        # Compute regular loss
        if isinstance(batch, (tuple, list)):
            target = batch[1]  # Assuming (input, target) format
        else:
            target = batch
            
        loss = nn.functional.cross_entropy(output, target)
        
        # Add safety penalty
        safety_penalty = self.safety_wrapper.compute_safety_penalty(output)
        total_loss = loss + self.config.get('safety_weight', 0.1) * safety_penalty
        
        return total_loss
        
    def update_safety_metrics(self, output: torch.Tensor):
        """Update safety metrics during training"""
        # Update metrics based on output
        if output.dim() == 1:  # For text outputs
            output_text = self.safety_wrapper._tensor_to_text(output)
            age_rating = self.safety_wrapper.safety_checker.get_age_rating(output_text)
            self.safety_metrics['age_rating'] = age_rating.value
        else:  # For image/other outputs
            # Update tensor-specific metrics
            self.safety_metrics['output_stats'] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'max': output.max().item(),
                'min': output.min().item()
            }
            
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Perform a single training step with safety checks"""
        # Validate batch
        batch = self.validate_batch(batch)
        
        # Forward pass
        if isinstance(batch, (tuple, list)):
            output = self.model(batch[0])
        else:
            output = self.model(batch)
            
        # Compute loss with safety penalty
        loss = self.compute_loss(output, batch)
        
        # Update safety metrics
        self.update_safety_metrics(output)
        
        return loss 