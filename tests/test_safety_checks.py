import pytest
from src.safety_checks import SafetyConfig, SafetyChecker, ModelSafetyWrapper, AgeRating

class MockModel:
    def generate(self, input_text: str, **kwargs):
        return input_text

def test_safety_checker():
    # Create a safety checker with custom configuration
    config = SafetyConfig(
        min_age_rating=AgeRating.TEEN,
        content_filters=["bad_word", "inappropriate"],
        max_input_length=100,
        block_sensitive_topics=True
    )
    checker = SafetyChecker(config)
    
    # Test input safety checks
    is_safe, reason = checker.check_input_safety("This is a normal input")
    assert is_safe
    assert reason == "Input passed safety checks"
    
    is_safe, reason = checker.check_input_safety("bad_word")
    assert not is_safe
    assert "content filter" in reason.lower()
    
    # Test output safety checks
    is_safe, reason = checker.check_output_safety("This is a normal output")
    assert is_safe
    
    is_safe, reason = checker.check_output_safety("This contains violence")
    assert not is_safe
    assert "sensitive topic" in reason.lower()

def test_age_rating():
    checker = SafetyChecker()
    
    # Test age rating determination
    assert checker.get_age_rating("Normal content") == AgeRating.ALL_AGES
    assert checker.get_age_rating("Teen content") == AgeRating.TEEN
    assert checker.get_age_rating("Violent content") == AgeRating.MATURE
    assert checker.get_age_rating("Explicit content") == AgeRating.ADULT

def test_model_safety_wrapper():
    # Create a mock model and safety checker
    model = MockModel()
    checker = SafetyChecker()
    wrapper = ModelSafetyWrapper(model, checker)
    
    # Test safe input/output
    output, safety_info = wrapper.generate("Safe input")
    assert safety_info["safety_check_passed"]
    assert safety_info["age_rating"] == AgeRating.ALL_AGES.value
    
    # Test unsafe input
    output, safety_info = wrapper.generate("This contains violence")
    assert not safety_info["safety_check_passed"]
    assert "safety check failed" in output.lower()

def test_content_warning():
    checker = SafetyChecker(SafetyConfig(require_content_warning=True))
    
    # Test content warning addition
    content = "This is mature content"
    warning_content = checker.add_content_warning(content, "MATURE")
    assert "[Content Warning: MATURE]" in warning_content
    
    # Test no warning for safe content
    content = "This is safe content"
    safe_content = checker.add_content_warning(content, "ALL_AGES")
    assert "[Content Warning" not in safe_content 