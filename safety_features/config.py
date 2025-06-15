from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SafetyConfig:
    min_age_rating: str
    content_filters: List[str]
    max_input_length: int
    max_output_length: int
    block_sensitive_topics: bool
    require_content_warning: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            "min_age_rating": {
                "name": self.min_age_rating,
                "value": self.min_age_rating
            },
            "content_filters": self.content_filters,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "block_sensitive_topics": self.block_sensitive_topics,
            "require_content_warning": self.require_content_warning
        } 