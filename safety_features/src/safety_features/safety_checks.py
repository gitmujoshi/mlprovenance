import re
from typing import Dict, List, Optional, Tuple, Any, Union
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

    def __str__(self):
        return self.value

    def to_dict(self):
        return {"name": self.name, "value": self.value}

class SafetyConfig:
    def __init__(
        self,
        min_age_rating: Union[str, AgeRating] = AgeRating.ALL_AGES,
        content_filters: Optional[List[str]] = None,
        max_input_length: int = 512,
        max_output_length: int = 100,
        block_sensitive_topics: bool = False,
        require_content_warning: bool = False
    ):
        self.min_age_rating = AgeRating[min_age_rating] if isinstance(min_age_rating, str) else min_age_rating
        self.content_filters = content_filters or []
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.block_sensitive_topics = block_sensitive_topics
        self.require_content_warning = require_content_warning

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            "min_age_rating": {
                "name": self.min_age_rating.name,
                "value": self.min_age_rating.value
            },
            "content_filters": self.content_filters,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "block_sensitive_topics": self.block_sensitive_topics,
            "require_content_warning": self.require_content_warning
        }

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
    def __init__(self, config: SafetyConfig):
        self.config = config
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
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "safety_config": self.config.to_dict(),
            "safety_metrics": self.metrics.to_dict(),
            "epoch_metrics": [{
                "epoch": 1,  # This will be updated by the training loop
                "total_checks": self.metrics.total_checks,
                "passed_checks": self.metrics.passed_checks,
                "content_warnings": self.metrics.content_warnings,
                "sensitive_topics": self.metrics.sensitive_topics_detected,
                "filter_violations": self.metrics.filter_violations,
                "losses": []  # This will be updated by the training loop
            }],
            "summary": {
                "total_checks": self.metrics.total_checks,
                "pass_rate": self.metrics.passed_checks / self.metrics.total_checks if self.metrics.total_checks > 0 else 0,
                "content_warnings": self.metrics.content_warnings,
                "most_common_sensitive_topics": sorted(
                    self.metrics.sensitive_topics_detected.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "most_common_filter_violations": sorted(
                    self.metrics.filter_violations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        }
    
    def save_safety_provenance(self, output_path: str):
        """
        Save safety provenance data to a JSON file.
        """
        provenance_data = self.get_safety_provenance()
        with open(output_path, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        logger.info(f"Safety provenance saved to {output_path}")

    def check_batch(self, batch: list) -> list:
        """
        Check a batch of texts for safety.
        Returns a list of safe texts.
        """
        safe_texts = []
        batch_metrics = {
            "total": len(batch),
            "passed": 0,
            "failed": 0,
            "reasons": {}
        }
        
        for text in batch:
            if not isinstance(text, str):
                logger.warning(f"Non-string item in batch: {type(text)}")
                batch_metrics["failed"] += 1
                batch_metrics["reasons"]["non_string"] = batch_metrics["reasons"].get("non_string", 0) + 1
                continue
                
            is_safe, safety_info = self.check_text(text)
            
            if is_safe:
                safe_texts.append(text)
                batch_metrics["passed"] += 1
            else:
                batch_metrics["failed"] += 1
                for reason in safety_info.get("filter_violations", []):
                    batch_metrics["reasons"][f"filter_{reason}"] = batch_metrics["reasons"].get(f"filter_{reason}", 0) + 1
                for topic in safety_info.get("sensitive_topics", []):
                    batch_metrics["reasons"][f"topic_{topic}"] = batch_metrics["reasons"].get(f"topic_{topic}", 0) + 1
        
        # Log batch metrics
        logger.info(f"Batch safety check: {batch_metrics['passed']}/{batch_metrics['total']} texts passed")
        if batch_metrics["failed"] > 0:
            logger.info(f"Failed reasons: {batch_metrics['reasons']}")
            
        return safe_texts

    def check_text(self, text: str) -> tuple[bool, dict]:
        """
        Check if text meets safety criteria.
        Returns a tuple of (is_safe, safety_info).
        """
        if not isinstance(text, str):
            return False, {"error": "Input must be a string"}
            
        safety_info = {
            "content_warning": False,
            "sensitive_topics": [],
            "filter_violations": []
        }
        
        # Check content filters
        for filter_name in self.content_filters:
            if self._check_content_filter(text, filter_name):
                safety_info["filter_violations"].append(filter_name)
                self.metrics.filter_violations[filter_name] = self.metrics.filter_violations.get(filter_name, 0) + 1
                
        # Check sensitive topics
        if self.config.block_sensitive_topics:
            for topic in self.sensitive_topics:
                if self._check_sensitive_topic(text, topic):
                    safety_info["sensitive_topics"].append(topic)
                    self.metrics.sensitive_topics_detected[topic] = self.metrics.sensitive_topics_detected.get(topic, 0) + 1
                    
        # Check content warning requirement
        if self.config.require_content_warning:
            safety_info["content_warning"] = True
            self.metrics.content_warnings += 1
            
        # Text is safe if it has no filter violations and no sensitive topics
        is_safe = len(safety_info["filter_violations"]) == 0 and len(safety_info["sensitive_topics"]) == 0
        
        # Update metrics
        self.metrics.total_checks += 1
        if is_safe:
            self.metrics.passed_checks += 1
        else:
            self.metrics.failed_checks += 1
        
        return is_safe, safety_info

    def _check_content_filter(self, text, filter_name):
        """Check if text matches a content filter."""
        return filter_name.lower() in text.lower()

    def _check_sensitive_topic(self, text, topic):
        """Check if text contains a sensitive topic."""
        return topic.lower() in text.lower()

class ModelSafetyWrapper:
    def __init__(self, model, safety_checker: SafetyChecker):
        self.model = model
        self.safety_checker = safety_checker
    
    def generate(self, input_text: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
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

    def check_batch(self, batch: list) -> list:
        """
        Check a batch of texts for safety.
        Returns a list of safe texts.
        """
        safe_texts = []
        for text in batch:
            try:
                is_safe, safety_info = self.safety_checker.check_text(text)
                if is_safe:
                    safe_texts.append(text)
                else:
                    logger.debug(f"Text failed safety check: {safety_info}")
            except Exception as e:
                logger.error(f"Error processing text in check_batch: {str(e)}")
                continue
                
        if not safe_texts:
            logger.warning(f"No safe items in batch {len(batch)}, skipping batch.")
            
        return safe_texts 