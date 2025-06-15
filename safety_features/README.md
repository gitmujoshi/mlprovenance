# Model Safety Features

This directory contains tools and utilities for adding safety checks to machine learning models. The safety features include:

- Age rating checks
- Content filtering
- Sensitive topic detection
- Content warnings
- Safety metrics tracking
- Provenance integration

## Setting Up the Environment

To set up the environment for the safety features, you can use the provided setup script. This script will create a virtual environment, install all necessary dependencies, and set up the package in development mode.

### Using the Setup Script

Run the following command in your terminal:

```bash
./scripts/setup.sh
```

This script will:
1. Create a virtual environment.
2. Activate the virtual environment.
3. Install all required packages.
4. Install the `safety_features` package in development mode.

After running the setup script, your environment will be ready to use the safety features.

## Structure

```
safety_features/
├── src/
│   ├── safety_checks.py     # Core safety checking functionality
│   └── train_with_safety.py # Training pipeline with safety checks
├── scripts/
│   ├── train_gpt2_with_safety.py  # Example using GPT-2
│   └── setup.sh             # Setup script for the environment
├── setup.py                 # Setup script for the package
└── README.md
```

## Installation

1. Install required packages:
```bash
pip install torch transformers datasets tqdm
```

2. Add the safety_features directory to your Python path:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

3. Install the safety_features package in development mode:
```bash
pip install -e . --use-pep517
```

### Understanding the Installation Command

The command `pip install -e . --use-pep517` is used to install the `safety_features` package in "editable" mode, which is particularly useful during development. Here's a detailed explanation of what this command does:

- **`pip install`**: This is the basic command to install a Python package using the `pip` package manager.

- **`-e`**: This flag stands for "editable" mode. When you install a package in editable mode, it creates a link to the source code instead of copying it to the site-packages directory. This means that any changes you make to the source code will be immediately reflected without needing to reinstall the package. This is especially useful during development when you are frequently modifying the code.

- **`.`**: This refers to the current directory, which should contain the `setup.py` or `pyproject.toml` file that defines the package and its dependencies.

- **`--use-pep517`**: This flag tells `pip` to use the PEP 517 build system for building the package. PEP 517 is a specification for a build system that allows for more flexibility and better integration with modern Python packaging tools. It uses the `pyproject.toml` file to define the build requirements and process, which can include specifying dependencies, build tools, and other metadata.

### What Happens During Installation

1. **Build System Activation**: When you run the command, `pip` activates the build system specified in the `pyproject.toml` file. This system is responsible for building the package and handling its dependencies.

2. **Dependency Resolution**: The build system reads the `pyproject.toml` file to determine the package's dependencies. It then checks if these dependencies are already installed in your environment. If not, it installs them.

3. **Editable Installation**: The package is installed in editable mode, which means that a link to the source code is created in the site-packages directory. This allows you to make changes to the source code and have them reflected immediately without reinstalling the package.

4. **Metadata and Configuration**: The build system also handles any additional metadata and configuration specified in the `pyproject.toml` file, such as package version, author information, and classifiers.

### Benefits

- **Development Efficiency**: Editable mode allows for rapid development and testing, as changes to the source code are immediately available without reinstalling the package.
- **Dependency Management**: The build system ensures that all necessary dependencies are installed, reducing the risk of missing or incompatible packages.
- **Modern Packaging**: Using PEP 517 allows for a more modern and flexible approach to Python packaging, which can be beneficial for complex projects.

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
./scripts/train_gpt2_with_safety.py \
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

## Training Dataset

The training script uses the WikiText-2 dataset, specifically the "wikitext-2-raw-v1" version, which is a collection of good-quality Wikipedia articles commonly used for language model training.

### Dataset Details
- **Source**: WikiText-2 dataset from Hugging Face
- **Version**: wikitext-2-raw-v1
- **Size**: Approximately 2 million tokens
- **Content**: Curated Wikipedia articles
- **Format**: Raw text with minimal preprocessing

### Data Storage
The dataset is automatically managed by the Hugging Face `datasets` library:
- **Default Location**: `~/.cache/huggingface/datasets`
- **Cache Management**: Automatically handled by the library
- **First-time Download**: Dataset is automatically downloaded when first run
- **Subsequent Runs**: Uses cached version if available

### Data Processing
The training pipeline includes the following preprocessing steps:
1. Loading the dataset using Hugging Face's `datasets` library
2. Filtering out empty or whitespace-only texts
3. Batching the data using PyTorch's DataLoader
4. Applying safety checks to each batch before training

### Safety Checks
Each batch of data goes through multiple safety checks:
- Content filtering
- Sensitive topic detection
- Age rating verification
- Content warning requirements
- Input/output length validation

## Running Training with Safety Features

To run training with safety features, follow these steps:

1. **Install the Package**: Make sure you have installed the `safety_features` package in development mode as described in the Installation section.

2. **Configure Safety Settings**: Use the `SafetyConfig` class to configure your safety settings. You can specify parameters such as `min_age_rating`, `content_filters`, `max_input_length`, and more.

3. **Wrap Your Model**: Use the `ModelSafetyWrapper` to wrap your model with safety checks. This ensures that all inputs and outputs are validated according to your safety configuration.

4. **Use the Training Pipeline**: Use the `SafeTrainingPipeline` to train your model with safety checks. This pipeline will automatically apply safety checks during training and track safety metrics.

5. **Run the Training Script**: Use the provided script or create your own to run the training process. For example, to run the GPT-2 example:

   ```bash
   ./scripts/train_gpt2_with_safety.py \
       --model-name gpt2 \
       --epochs 3 \
       --batch-size 32 \
       --min-age-rating TEEN \
       --max-input-length 512 \
       --content-filters "bad_word" "inappropriate" \
       --block-sensitive-topics \
       --require-content-warning
   ```

6. **Review Safety Reports**: After training, review the safety reports generated in the `artifacts/provenance` directory to analyze the safety metrics and any issues encountered during training.

7. **Run Artifacts**: Each training run creates a unique folder (named with a timestamp) under the output directory (default: `artifacts/provenance`). This folder contains:
   - The provenance report
   - The Merkle tree JSON file
   - The trained model (saved as "trained_model.pt")

   This structure ensures that all artifacts from a single run are kept together, making it easier to track and manage different training runs.

## Testing Safety Features with a Sample Dataset

To validate that the safety features are working as expected, you can generate a synthetic test dataset containing a mix of appropriate and inappropriate content, and then run automated safety checks on it.

### 1. Generate a Test Dataset

Use the provided script to create a dataset with both safe and unsafe examples:

```bash
./scripts/generate_test_dataset.py
```

- This will create a file at `artifacts/test_dataset.json` containing 50 samples with a variety of content types (good, bad words, sensitive topics, inappropriate, long, etc.).
- Each sample is annotated with the expected safety check result for validation.

### 2. Test Safety Features on the Dataset

Run the following script to evaluate the safety features against the generated dataset:

```bash
./scripts/test_safety_features.py
```

- This will process each sample, apply the safety checks, and compare the results to the expected outcomes.
- A detailed report will be saved to `artifacts/safety_test_report.json`.
- The report includes summary statistics (accuracy, pass/fail counts) and per-sample results.

#### Example Output
```
INFO:__main__:Safety feature testing completed:
INFO:__main__:Total samples: 50
INFO:__main__:Passed checks: 47
INFO:__main__:Failed checks: 3
INFO:__main__:Accuracy: 94.00%
INFO:__main__:Detailed report saved to: artifacts/safety_test_report.json
```

### Why Use This?
- **Regression Testing:** Ensure that changes to safety logic do not break expected behavior.
- **Demonstration:** Show that inappropriate or unsafe data is filtered out as intended.
- **Customization:** Easily extend the test dataset or safety logic for your use case.

--- 