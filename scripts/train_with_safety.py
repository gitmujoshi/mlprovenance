#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.safety_checks import SafetyConfig, AgeRating
from src.train_with_safety import SafeTrainingPipeline
from src.provenance import ProvenanceTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train model with safety checks')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='artifacts/provenance',
                      help='Directory to save provenance and safety reports')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--min-age-rating', type=str, default='TEEN',
                      choices=['ALL_AGES', 'TEEN', 'MATURE', 'ADULT'],
                      help='Minimum age rating for content')
    parser.add_argument('--max-input-length', type=int, default=1000,
                      help='Maximum input length')
    parser.add_argument('--max-output-length', type=int, default=2000,
                      help='Maximum output length')
    parser.add_argument('--content-filters', type=str, nargs='+',
                      default=['bad_word', 'inappropriate'],
                      help='List of content filters to apply')
    parser.add_argument('--block-sensitive-topics', action='store_true',
                      help='Enable sensitive topic blocking')
    parser.add_argument('--require-content-warning', action='store_true',
                      help='Require content warnings for mature content')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure safety settings
    safety_config = SafetyConfig(
        min_age_rating=AgeRating[args.min_age_rating],
        content_filters=args.content_filters,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        block_sensitive_topics=args.block_sensitive_topics,
        require_content_warning=args.require_content_warning
    )
    
    # Initialize your model here
    # Replace this with your actual model initialization
    from src.model import YourModel  # Import your actual model
    model = YourModel()
    
    # Create training pipeline with safety checks
    pipeline = SafeTrainingPipeline(
        model=model,
        safety_config=safety_config,
        provenance_dir=str(output_dir)
    )
    
    # Load your training data
    # Replace this with your actual data loading code
    from src.data import load_training_data  # Import your data loading function
    train_data = load_training_data(args.data_dir)
    
    try:
        # Train with safety checks
        logger.info("Starting training with safety checks...")
        pipeline.train(
            train_data=train_data,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Get and print safety report
        safety_report = pipeline.get_safety_report()
        logger.info("\nSafety Report Summary:")
        logger.info(f"Total Safety Checks: {safety_report['summary']['total_checks']}")
        logger.info(f"Safety Check Pass Rate: {safety_report['summary']['pass_rate']:.2%}")
        logger.info(f"Content Warnings Added: {safety_report['summary']['content_warnings']}")
        
        logger.info("\nMost Common Sensitive Topics:")
        for topic, count in safety_report['summary']['most_common_sensitive_topics']:
            logger.info(f"  - {topic}: {count}")
        
        logger.info("\nMost Common Filter Violations:")
        for filter_pattern, count in safety_report['summary']['most_common_filter_violations']:
            logger.info(f"  - {filter_pattern}: {count}")
        
        logger.info(f"\nDetailed safety report saved to: {output_dir}/safety_provenance.json")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 