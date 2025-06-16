from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..safety_checks import SafetyChecker, SafetyConfig

class BaseSafetyWrapper(ABC):
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.safety_checker = SafetyChecker(config)
        
    @abstractmethod
    def validate_input(self, input_data: Any) -> Any:
        """Validate input data for safety"""
        pass
        
    @abstractmethod
    def validate_output(self, output_data: Any) -> Any:
        """Validate output data for safety"""
        pass
        
    @abstractmethod
    def compute_safety_penalty(self, output_data: Any) -> float:
        """Compute safety violation penalty"""
        pass
        
    def get_metrics(self) -> Dict:
        """Get safety metrics"""
        return self.safety_checker.get_safety_provenance()

class BaseSafetyTrainer(ABC):
    def __init__(self, model: Any, config: Dict):
        self.model = model
        self.config = config
        self.safety_metrics = {}
        
    @abstractmethod
    def validate_batch(self, batch: Any) -> Any:
        """Validate training batch for safety"""
        pass
        
    @abstractmethod
    def compute_loss(self, output: Any, batch: Any) -> float:
        """Compute training loss"""
        pass
        
    @abstractmethod
    def update_safety_metrics(self, output: Any):
        """Update safety metrics"""
        pass
        
    def get_safety_metrics(self) -> Dict:
        """Get safety metrics"""
        return self.safety_metrics 