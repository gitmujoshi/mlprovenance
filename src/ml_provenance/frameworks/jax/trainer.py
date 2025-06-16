import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
from ..base import BaseSafetyTrainer
from .wrapper import JAXSafetyWrapper

class JAXSafetyTrainer(BaseSafetyTrainer):
    def __init__(self, model: Any, config: Dict):
        super().__init__(model, config)
        self.safety_wrapper = JAXSafetyWrapper(config.get('safety_config'))
        
    def validate_batch(self, batch: jnp.ndarray) -> jnp.ndarray:
        """Validate training batch for safety"""
        if isinstance(batch, (tuple, list)):
            # Handle multiple inputs (e.g., input and target)
            return tuple(self.safety_wrapper.validate_input(x) for x in batch)
        return self.safety_wrapper.validate_input(batch)
        
    def compute_loss(self, output: jnp.ndarray, batch: jnp.ndarray) -> jnp.ndarray:
        """Compute training loss with safety penalty"""
        # Compute regular loss
        if isinstance(batch, (tuple, list)):
            target = batch[1]  # Assuming (input, target) format
        else:
            target = batch
            
        loss = jnp.mean(-jnp.sum(target * jnp.log(output + 1e-10), axis=-1))
        
        # Add safety penalty
        safety_penalty = self.safety_wrapper.compute_safety_penalty(output)
        total_loss = loss + self.config.get('safety_weight', 0.1) * safety_penalty
        
        return total_loss
        
    def update_safety_metrics(self, output: jnp.ndarray):
        """Update safety metrics during training"""
        # Update metrics based on output
        if len(output.shape) == 1:  # For text outputs
            output_text = self.safety_wrapper._array_to_text(output)
            age_rating = self.safety_wrapper.safety_checker.get_age_rating(output_text)
            self.safety_metrics['age_rating'] = age_rating.value
        else:  # For image/other outputs
            # Update array-specific metrics
            self.safety_metrics['output_stats'] = {
                'mean': jnp.mean(output),
                'std': jnp.std(output),
                'max': jnp.max(output),
                'min': jnp.min(output)
            }
            
    @jax.jit
    def train_step(self, state: Any, batch: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """Perform a single training step with safety checks"""
        def loss_fn(params):
            # Validate batch
            batch = self.validate_batch(batch)
            
            # Forward pass
            if isinstance(batch, (tuple, list)):
                output = self.model.apply(params, batch[0], training=True)
            else:
                output = self.model.apply(params, batch, training=True)
                
            # Compute loss with safety penalty
            return self.compute_loss(output, batch)
            
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        # Update model parameters
        state = state.apply_gradients(grads=grads)
        
        # Update safety metrics
        self.update_safety_metrics(output)
        
        return state, loss 