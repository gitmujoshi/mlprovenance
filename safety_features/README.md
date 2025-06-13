# Model Safety Features

This directory contains tools and utilities for adding safety checks to machine learning models. The safety features include:

- Age rating checks
- Content filtering
- Sensitive topic detection
- Content warnings
- Safety metrics tracking
- Provenance integration

## Structure

```
safety_features/
├── src/
│   ├── safety_checks.py     # Core safety checking functionality
│   └── train_with_safety.py # Training pipeline with safety checks
├── scripts/
│   └── train_gpt2_with_safety.py  # Example using GPT-2
└── README.md
```

## Installation

1. Install required packages:
```bash
pip install torch transformers datasets tqdm
```

2. Add the safety_features directory to your Python path:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/safety_features
```

## Usage

### Basic Usage

```python
from safety_features.src.safety_checks import SafetyConfig, SafetyChecker, ModelSafetyWrapper
from safety_features.src.train_with_safety import SafeTrainingPipeline

# Configure safety settings
safety_config = SafetyConfig(
    min_age_rating=AgeRating.TEEN,
    content_filters=["bad_word", "inappropriate"],
    max_input_length=1000,
    block_sensitive_topics=True
)

# Create safety checker
safety_checker = SafetyChecker(safety_config)

# Wrap your model
safe_model = ModelSafetyWrapper(your_model, safety_checker)

# Use in training pipeline
pipeline = SafeTrainingPipeline(
    model=your_model,
    safety_config=safety_config
)
```

### Running the GPT-2 Example

```bash
python safety_features/scripts/train_gpt2_with_safety.py \
    --model-name gpt2 \
    --epochs 3 \
    --batch-size 32 \
    --min-age-rating TEEN \
    --max-input-length 512 \
    --content-filters "bad_word" "inappropriate" \
    --block-sensitive-topics \
    --require-content-warning
```

## Features

### Safety Checks
- Input/output length validation
- Age rating determination
- Content filtering
- Sensitive topic detection
- Content warning management

### Metrics Tracking
- Total safety checks
- Pass/fail rates
- Age rating distribution
- Content warning counts
- Sensitive topic statistics
- Filter violation tracking

### Provenance Integration
- Safety configuration tracking
- Per-epoch safety metrics
- Final safety report
- Error tracking

## Customization

You can customize the safety checks by:

1. Modifying the `SafetyConfig` parameters
2. Adding new content filters
3. Extending the sensitive topics list
4. Implementing custom age rating logic
5. Adding new safety metrics

## Contributing

Feel free to:
1. Add new safety features
2. Improve existing checks
3. Add support for more models
4. Enhance the metrics tracking
5. Add visualization tools

## Latest Updates

- **Data Preprocessing**: The training script now filters out empty lines from the dataset to avoid errors during batching.
- **Generation Parameters**: The `generate` call now uses `max_new_tokens` instead of `max_length` to avoid input length warnings. 