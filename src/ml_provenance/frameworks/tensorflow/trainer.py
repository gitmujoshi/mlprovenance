import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from ..base import BaseSafetyTrainer
from .wrapper import TensorFlowSafetyWrapper

class TensorFlowSafetyTrainer(BaseSafetyTrainer):
    def __init__(self, model: tf.keras.Model, config: Dict):
        super().__init__(model, config)
        self.safety_wrapper = TensorFlowSafetyWrapper(config.get('safety_config'))
        
    def validate_batch(self, batch: tf.Tensor) -> tf.Tensor:
        """Validate training batch for safety"""
        if isinstance(batch, (tuple, list)):
            # Handle multiple inputs (e.g., input and target)
            return tuple(self.safety_wrapper.validate_input(x) for x in batch)
        return self.safety_wrapper.validate_input(batch)
        
    def compute_loss(self, output: tf.Tensor, batch: tf.Tensor) -> tf.Tensor:
        """Compute training loss with safety penalty"""
        # Compute regular loss
        if isinstance(batch, (tuple, list)):
            target = batch[1]  # Assuming (input, target) format
        else:
            target = batch
            
        loss = tf.keras.losses.categorical_crossentropy(target, output)
        
        # Add safety penalty
        safety_penalty = self.safety_wrapper.compute_safety_penalty(output)
        total_loss = loss + self.config.get('safety_weight', 0.1) * safety_penalty
        
        return total_loss
        
    def update_safety_metrics(self, output: tf.Tensor):
        """Update safety metrics during training"""
        # Update metrics based on output
        if len(output.shape) == 1:  # For text outputs
            output_text = self.safety_wrapper._tensor_to_text(output)
            age_rating = self.safety_wrapper.safety_checker.get_age_rating(output_text)
            self.safety_metrics['age_rating'] = age_rating.value
        else:  # For image/other outputs
            # Update tensor-specific metrics
            self.safety_metrics['output_stats'] = {
                'mean': tf.reduce_mean(output).numpy(),
                'std': tf.math.reduce_std(output).numpy(),
                'max': tf.reduce_max(output).numpy(),
                'min': tf.reduce_min(output).numpy()
            }
            
    @tf.function
    def train_step(self, batch: tf.Tensor) -> tf.Tensor:
        """Perform a single training step with safety checks"""
        with tf.GradientTape() as tape:
            # Validate batch
            batch = self.validate_batch(batch)
            
            # Forward pass
            if isinstance(batch, (tuple, list)):
                output = self.model(batch[0], training=True)
            else:
                output = self.model(batch, training=True)
                
            # Compute loss with safety penalty
            loss = self.compute_loss(output, batch)
            
        # Update model weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update safety metrics
        self.update_safety_metrics(output)
        
        return loss 