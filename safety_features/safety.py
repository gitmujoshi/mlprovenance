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
    sensitive_topics: bool
    inappropriate_content: bool

class SafetyChecker:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.bad_words = set([
            "damn", "hell", "crap", "stupid", "idiot",  # Add more as needed
        ])
        self.sensitive_topics = set([
            "politics", "religion", "sex", "drugs", "violence",  # Add more as needed
        ])
        self.inappropriate_patterns = [
            r"inappropriate",
            r"offensive",
            r"explicit",
            # Add more patterns as needed
        ]
    
    def check_text(self, text: str) -> Dict[str, Any]:
        """Check text against safety criteria and return results."""
        warnings = []
        violations = []
        
        # Check text length
        if len(text) > self.config.max_input_length:
            warnings.append(f"Text exceeds maximum input length of {self.config.max_input_length}")
            violations.append("length_violation")
        
        # Check for bad words
        found_bad_words = [word for word in self.bad_words if word.lower() in text.lower()]
        if found_bad_words:
            warnings.append(f"Found inappropriate words: {', '.join(found_bad_words)}")
            violations.append("bad_words")
        
        # Check for sensitive topics
        found_sensitive_topics = [topic for topic in self.sensitive_topics if topic.lower() in text.lower()]
        if found_sensitive_topics:
            warnings.append(f"Found sensitive topics: {', '.join(found_sensitive_topics)}")
            violations.append("sensitive_topics")
        
        # Check for inappropriate patterns
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text.lower()):
                warnings.append(f"Found inappropriate content matching pattern: {pattern}")
                violations.append("inappropriate_content")
        
        # Determine if content warning is needed
        content_warning = bool(warnings) and self.config.require_content_warning
        
        # Determine if sensitive topics are present
        sensitive_topics = "sensitive_topics" in violations
        
        # Determine if inappropriate content is present
        inappropriate_content = any(v in violations for v in ["bad_words", "inappropriate_content"])
        
        # Determine if all checks pass
        passes_checks = not violations
        
        return {
            "passes_checks": passes_checks,
            "warnings": warnings,
            "violations": violations,
            "content_warning": content_warning,
            "sensitive_topics": sensitive_topics,
            "inappropriate_content": inappropriate_content
        } 