import re
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AgeRating(Enum):
    ALL_AGES = "ALL_AGES"
    TEEN = "TEEN"
    MATURE = "MATURE"
    ADULT = "ADULT"

@dataclass
class SafetyConfig:
    min_age_rating: AgeRating = AgeRating.ALL_AGES
    content_filters: List[str] = None
    max_input_length: int = 1000
    max_output_length: int = 2000
    block_sensitive_topics: bool = True
    require_content_warning: bool = False

@dataclass
class SafetyMetrics:
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    age_ratings: Dict[str, int] = None
    content_warnings: int = 0
    sensitive_topics_detected: Dict[str, int] = None
    filter_violations: Dict[str, int] = None
    
    def __post_init__(self):
        if self.age_ratings is None:
            self.age_ratings = {rating.value: 0 for rating in AgeRating}
        if self.sensitive_topics_detected is None:
            self.sensitive_topics_detected = {}
        if self.filter_violations is None:
            self.filter_violations = {}
    
    def to_dict(self) -> Dict:
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "pass_rate": self.passed_checks / self.total_checks if self.total_checks > 0 else 0,
            "age_ratings": self.age_ratings,
            "content_warnings": self.content_warnings,
            "sensitive_topics_detected": self.sensitive_topics_detected,
            "filter_violations": self.filter_violations
        }

class SafetyChecker:
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.sensitive_topics = [
            "violence", "explicit", "harmful", "illegal",
            "discrimination", "harassment", "self-harm"
        ]
        self.content_filters = self.config.content_filters or []
        self.metrics = SafetyMetrics()
        
    def check_input_safety(self, input_text: str) -> Tuple[bool, str]:
        """
        Check if the input text meets safety requirements.
        Returns (is_safe, reason)
        """
        self.metrics.total_checks += 1
        
        # Check input length
        if len(input_text) > self.config.max_input_length:
            self.metrics.failed_checks += 1
            return False, f"Input exceeds maximum length of {self.config.max_input_length} characters"
        
        # Check for sensitive topics
        if self.config.block_sensitive_topics:
            for topic in self.sensitive_topics:
                if re.search(rf'\b{topic}\b', input_text.lower()):
                    self.metrics.failed_checks += 1
                    self.metrics.sensitive_topics_detected[topic] = self.metrics.sensitive_topics_detected.get(topic, 0) + 1
                    return False, f"Input contains sensitive topic: {topic}"
        
        # Check content filters
        for filter_pattern in self.content_filters:
            if re.search(filter_pattern, input_text, re.IGNORECASE):
                self.metrics.failed_checks += 1
                self.metrics.filter_violations[filter_pattern] = self.metrics.filter_violations.get(filter_pattern, 0) + 1
                return False, f"Input matches content filter: {filter_pattern}"
        
        self.metrics.passed_checks += 1
        return True, "Input passed safety checks"
    
    def check_output_safety(self, output_text: str) -> Tuple[bool, str]:
        """
        Check if the model output meets safety requirements.
        Returns (is_safe, reason)
        """
        self.metrics.total_checks += 1
        
        # Check output length
        if len(output_text) > self.config.max_output_length:
            self.metrics.failed_checks += 1
            return False, f"Output exceeds maximum length of {self.config.max_output_length} characters"
        
        # Check for sensitive topics
        if self.config.block_sensitive_topics:
            for topic in self.sensitive_topics:
                if re.search(rf'\b{topic}\b', output_text.lower()):
                    self.metrics.failed_checks += 1
                    self.metrics.sensitive_topics_detected[topic] = self.metrics.sensitive_topics_detected.get(topic, 0) + 1
                    return False, f"Output contains sensitive topic: {topic}"
        
        # Check content filters
        for filter_pattern in self.content_filters:
            if re.search(filter_pattern, output_text, re.IGNORECASE):
                self.metrics.failed_checks += 1
                self.metrics.filter_violations[filter_pattern] = self.metrics.filter_violations.get(filter_pattern, 0) + 1
                return False, f"Output matches content filter: {filter_pattern}"
        
        self.metrics.passed_checks += 1
        return True, "Output passed safety checks"
    
    def get_age_rating(self, content: str) -> AgeRating:
        """
        Determine the age rating for the given content.
        """
        # Simple implementation - can be made more sophisticated
        if any(topic in content.lower() for topic in ["explicit", "adult", "mature"]):
            rating = AgeRating.ADULT
        elif any(topic in content.lower() for topic in ["violence", "harmful"]):
            rating = AgeRating.MATURE
        elif any(topic in content.lower() for topic in ["teen", "adolescent"]):
            rating = AgeRating.TEEN
        else:
            rating = AgeRating.ALL_AGES
            
        self.metrics.age_ratings[rating.value] += 1
        return rating
    
    def add_content_warning(self, content: str, warning_type: str) -> str:
        """
        Add a content warning to the content if required.
        """
        if self.config.require_content_warning:
            warning = f"[Content Warning: {warning_type}]\n\n"
            self.metrics.content_warnings += 1
            return warning + content
        return content
    
    def get_safety_provenance(self) -> Dict:
        """
        Generate safety provenance data for the training cycle.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "safety_config": {
                "min_age_rating": self.config.min_age_rating.value,
                "content_filters": self.content_filters,
                "max_input_length": self.config.max_input_length,
                "max_output_length": self.config.max_output_length,
                "block_sensitive_topics": self.config.block_sensitive_topics,
                "require_content_warning": self.config.require_content_warning
            },
            "metrics": self.metrics.to_dict(),
            "sensitive_topics": self.sensitive_topics
        }
    
    def save_safety_provenance(self, output_path: str):
        """
        Save safety provenance data to a JSON file.
        """
        provenance_data = self.get_safety_provenance()
        with open(output_path, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        logger.info(f"Safety provenance saved to {output_path}")

class ModelSafetyWrapper:
    def __init__(self, model, safety_checker: SafetyChecker):
        self.model = model
        self.safety_checker = safety_checker
    
    def generate(self, input_text: str, **kwargs) -> Tuple[str, Dict]:
        """
        Generate output with safety checks.
        Returns (output_text, safety_info)
        """
        # Check input safety
        is_safe, reason = self.safety_checker.check_input_safety(input_text)
        if not is_safe:
            return f"Safety check failed: {reason}", {"safety_check_passed": False, "reason": reason}
        
        # Generate output
        output = self.model.generate(input_text, **kwargs)
        
        # Check output safety
        is_safe, reason = self.safety_checker.check_output_safety(output)
        if not is_safe:
            return f"Safety check failed: {reason}", {"safety_check_passed": False, "reason": reason}
        
        # Get age rating
        age_rating = self.safety_checker.get_age_rating(output)
        
        # Add content warning if needed
        if age_rating in [AgeRating.MATURE, AgeRating.ADULT]:
            output = self.safety_checker.add_content_warning(output, age_rating.value)
        
        safety_info = {
            "safety_check_passed": True,
            "age_rating": age_rating.value,
            "content_warning_added": age_rating in [AgeRating.MATURE, AgeRating.ADULT]
        }
        
        return output, safety_info 