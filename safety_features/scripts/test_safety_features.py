import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from safety_features.safety import SafetyChecker
from safety_features.config import SafetyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_dataset(file_path: str) -> Dict[str, Any]:
    """Load the test dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def test_safety_features(dataset_path: str, output_path: str):
    """Test safety features on the test dataset and generate a report."""
    # Load dataset
    dataset = load_test_dataset(dataset_path)
    
    # Initialize safety checker with default config
    safety_config = SafetyConfig(
        min_age_rating="TEEN",
        content_filters=["bad_word", "inappropriate"],
        max_input_length=512,
        max_output_length=100,
        block_sensitive_topics=True,
        require_content_warning=True
    )
    safety_checker = SafetyChecker(safety_config)
    
    # Test each sample
    results = []
    total_samples = len(dataset["dataset"])
    passed_checks = 0
    failed_checks = 0
    
    for i, sample in enumerate(dataset["dataset"], 1):
        text = sample["text"]
        expected = sample["expected_safety"]
        
        # Run safety checks
        safety_result = safety_checker.check_text(text)
        
        # Compare with expected results
        actual_passes = safety_result["passes_checks"]
        expected_passes = expected["passes_checks"]
        
        if actual_passes == expected_passes:
            passed_checks += 1
        else:
            failed_checks += 1
        
        results.append({
            "sample_id": i,
            "text": text,
            "expected_safety": expected,
            "actual_safety": safety_result,
            "matches_expected": actual_passes == expected_passes,
            "warnings": safety_result.get("warnings", []),
            "violations": safety_result.get("violations", [])
        })
    
    # Generate report
    report = {
        "summary": {
            "total_samples": total_samples,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "accuracy": (passed_checks / total_samples) * 100
        },
        "expected_distribution": dataset["metadata"]["expected_distribution"],
        "detailed_results": results
    }
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info(f"Safety feature testing completed:")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Passed checks: {passed_checks}")
    logger.info(f"Failed checks: {failed_checks}")
    logger.info(f"Accuracy: {report['summary']['accuracy']:.2f}%")
    logger.info(f"Detailed report saved to: {output_path}")

if __name__ == "__main__":
    dataset_path = "artifacts/test_dataset.json"
    output_path = "artifacts/safety_test_report.json"
    test_safety_features(dataset_path, output_path) 