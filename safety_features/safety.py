import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .config import SafetyConfig

@dataclass
class SafetyResult:
    passes_checks: bool
    warnings: List[str]
    violations: List[str]
    content_warning: bool
    sensitive_topics: List[str]
    inappropriate_content: bool

class SafetyChecker:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.bad_words = {"bad", "inappropriate", "explicit"}
        self.sensitive_topics = {"violence", "drugs", "politics"}
        self.inappropriate_patterns = [
            r"explicit content",
            r"inappropriate material",
            r"sensitive topic"
        ]

    def check_text(self, text: str) -> Dict[str, Any]:
        warnings = []
        violations = []
        
        # Check text length
        if len(text) > self.config.max_input_length:
            warnings.append(f"Text exceeds maximum length of {self.config.max_input_length} characters")
        
        # Check for bad words
        for word in self.bad_words:
            if word in text.lower():
                violations.append(f"Contains inappropriate word: {word}")
        
        # Check for sensitive topics
        if self.config.block_sensitive_topics:
            for topic in self.sensitive_topics:
                if topic in text.lower():
                    violations.append(f"Contains sensitive topic: {topic}")
        
        # Check for inappropriate patterns
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text.lower()):
                violations.append(f"Matches inappropriate pattern: {pattern}")
        
        # Determine if content warning is needed
        content_warning = bool(violations) and self.config.require_content_warning
        
        # Get list of sensitive topics found
        found_topics = [topic for topic in self.sensitive_topics if topic in text.lower()]
        
        return {
            "passes_checks": not violations,
            "warnings": warnings,
            "violations": violations,
            "content_warning": content_warning,
            "sensitive_topics": found_topics,
            "inappropriate_content": bool(violations)
        } 