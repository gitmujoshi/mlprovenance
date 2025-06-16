import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from ..base import BaseSafetyWrapper
from ...safety_checks import SafetyConfig

class TensorFlowSafetyWrapper(BaseSafetyWrapper):
    def __init__(self, config: Optional[SafetyConfig] = None):
        super().__init__(config)
        
    def validate_input(self, input_data: tf.Tensor) -> tf.Tensor:
        """Validate TensorFlow input tensor for safety"""
        # Convert tensor to string for text-based safety checks
        if len(input_data.shape) == 1:  # For text inputs
            input_text = self._tensor_to_text(input_data)
            is_safe, _ = self.safety_checker.check_input_safety(input_text)
            if not is_safe:
                raise tf.errors.InvalidArgumentError(
                    None, None, "Input failed safety checks")
        else:  # For image/other inputs
            self._validate_tensor(input_data)
        return input_data
        
    def validate_output(self, output_data: tf.Tensor) -> tf.Tensor:
        """Validate TensorFlow output tensor for safety"""
        # Convert tensor to string for text-based safety checks
        if len(output_data.shape) == 1:  # For text outputs
            output_text = self._tensor_to_text(output_data)
            is_safe, _ = self.safety_checker.check_output_safety(output_text)
            if not is_safe:
                raise tf.errors.InvalidArgumentError(
                    None, None, "Output failed safety checks")
        else:  # For image/other outputs
            self._validate_tensor(output_data)
        return output_data
        
    def compute_safety_penalty(self, output_data: tf.Tensor) -> tf.Tensor:
        """Compute safety violation penalty for TensorFlow tensors"""
        penalty = tf.constant(0.0, dtype=output_data.dtype)
        
        # Add penalties for different safety violations
        if len(output_data.shape) == 1:  # For text outputs
            output_text = self._tensor_to_text(output_data)
            age_rating = self.safety_checker.get_age_rating(output_text)
            if age_rating.value in ["MATURE", "ADULT"]:
                penalty += 0.1
        else:  # For image/other outputs
            # Add penalties for tensor-specific violations
            if tf.reduce_any(tf.math.is_nan(output_data)):
                penalty += 0.2
            if tf.reduce_any(tf.math.is_inf(output_data)):
                penalty += 0.2
                
        return penalty
        
    def _validate_tensor(self, tensor: tf.Tensor):
        """Validate tensor properties"""
        if tf.reduce_any(tf.math.is_nan(tensor)):
            raise tf.errors.InvalidArgumentError(
                None, None, "Tensor contains NaN values")
        if tf.reduce_any(tf.math.is_inf(tensor)):
            raise tf.errors.InvalidArgumentError(
                None, None, "Tensor contains Inf values")
                
    def _tensor_to_text(self, tensor: tf.Tensor) -> str:
        """Convert tensor to text for safety checks"""
        # Implement tensor to text conversion based on your model's tokenizer
        # This is a placeholder implementation
        return " ".join(str(x.numpy()) for x in tensor) 