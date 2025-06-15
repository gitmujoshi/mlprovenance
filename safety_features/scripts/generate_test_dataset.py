import json
import random
from pathlib import Path
from datetime import datetime

def generate_test_dataset(output_file: str, num_samples: int = 50):
    """
    Generate a test dataset with a mix of appropriate and inappropriate content.
    Each sample will be tagged with expected safety check results.
    """
    # Sample texts with different characteristics
    samples = [
        # Appropriate content
        {
            "text": "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing.",
            "expected_safety": {
                "passes_checks": True,
                "content_warning": False,
                "sensitive_topics": False,
                "inappropriate_content": False
            }
        },
        {
            "text": "The weather is beautiful today. I went for a walk in the park and saw many flowers blooming.",
            "expected_safety": {
                "passes_checks": True,
                "content_warning": False,
                "sensitive_topics": False,
                "inappropriate_content": False
            }
        },
        {
            "text": "I love learning about science and technology. The latest developments in AI are fascinating.",
            "expected_safety": {
                "passes_checks": True,
                "content_warning": False,
                "sensitive_topics": False,
                "inappropriate_content": False
            }
        },
        
        # Content with bad words
        {
            "text": "This is a test with some bad words like damn and hell. We should filter these out.",
            "expected_safety": {
                "passes_checks": False,
                "content_warning": True,
                "sensitive_topics": False,
                "inappropriate_content": True
            }
        },
        
        # Content with sensitive topics
        {
            "text": "This text discusses sensitive topics like politics and religion. It should be flagged.",
            "expected_safety": {
                "passes_checks": False,
                "content_warning": True,
                "sensitive_topics": True,
                "inappropriate_content": False
            }
        },
        
        # Content with inappropriate themes
        {
            "text": "This content contains inappropriate themes and should be filtered out by the safety checks.",
            "expected_safety": {
                "passes_checks": False,
                "content_warning": True,
                "sensitive_topics": False,
                "inappropriate_content": True
            }
        },
        
        # Very long content
        {
            "text": "This is a very long text that exceeds the maximum input length limit. " * 100,
            "expected_safety": {
                "passes_checks": False,
                "content_warning": True,
                "sensitive_topics": False,
                "inappropriate_content": False
            }
        },
        
        # Mixed content
        {
            "text": "This is a normal sentence followed by some bad words and sensitive topics. It should fail multiple checks.",
            "expected_safety": {
                "passes_checks": False,
                "content_warning": True,
                "sensitive_topics": True,
                "inappropriate_content": True
            }
        }
    ]
    
    # Generate the dataset
    dataset = []
    for _ in range(num_samples):
        # Randomly select a sample and create variations
        base_sample = random.choice(samples)
        text = base_sample["text"]
        
        # Add some random variations to the text
        if random.random() < 0.3:  # 30% chance to modify the text
            text = text + " " + random.choice([
                "Additional context here.",
                "More information to consider.",
                "Some extra details.",
                "Further explanation needed."
            ])
        
        dataset.append({
            "text": text,
            "expected_safety": base_sample["expected_safety"],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sample_type": "test",
                "version": "1.0"
            }
        })
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    with open(output_file, 'w') as f:
        json.dump({
            "dataset": dataset,
            "metadata": {
                "total_samples": len(dataset),
                "generated_at": datetime.now().isoformat(),
                "description": "Test dataset for safety feature validation",
                "expected_distribution": {
                    "passes_checks": sum(1 for s in dataset if s["expected_safety"]["passes_checks"]),
                    "fails_checks": sum(1 for s in dataset if not s["expected_safety"]["passes_checks"]),
                    "content_warnings": sum(1 for s in dataset if s["expected_safety"]["content_warning"]),
                    "sensitive_topics": sum(1 for s in dataset if s["expected_safety"]["sensitive_topics"]),
                    "inappropriate_content": sum(1 for s in dataset if s["expected_safety"]["inappropriate_content"])
                }
            }
        }, f, indent=2)
    
    print(f"Generated test dataset with {len(dataset)} samples")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    output_file = "artifacts/test_dataset.json"
    generate_test_dataset(output_file) 