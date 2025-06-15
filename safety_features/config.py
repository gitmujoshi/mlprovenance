from dataclasses import dataclass
from typing import List
from enum import Enum

class AgeRating(Enum):
    CHILD = "CHILD"
    TEEN = "TEEN"
    ADULT = "ADULT"

@dataclass
class SafetyConfig:
    min_age_rating: AgeRating
    content_filters: List[str]
    max_input_length: int
    max_output_length: int
    block_sensitive_topics: bool = True
    require_content_warning: bool = True

    def to_dict(self) -> dict:
        return {
            "min_age_rating": self.min_age_rating.value,
            "content_filters": self.content_filters,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "block_sensitive_topics": self.block_sensitive_topics,
            "require_content_warning": self.require_content_warning
        } 