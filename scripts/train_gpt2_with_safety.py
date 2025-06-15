#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.safety_checks import SafetyConfig, AgeRating
from src.train_with_safety import SafeTrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2Model:
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.train()
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate(self, input_text: str, **kwargs):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=kwargs.get('max_length', 100),
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def update(self, texts):
        # Prepare batch
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Forward pass
        outputs = self.model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update weights (in a real training loop, you would use an optimizer here)
        return loss.item()

def load_training_data(data_dir=None):
    """
    Load a curated dataset for training.
    We'll use the 'wikitext' dataset as an example, but you can replace this with your preferred dataset.
    """
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Preprocess data
    def preprocess_function(examples):
        return {"text": examples["text"]}
    
    processed_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        batched=True
    )
    
    # Create dataloader
    def collate_fn(batch):
        return [item["text"] for item in batch]
    
    dataloader = DataLoader(
        processed_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 with safety checks')
    parser.add_argument('--model-name', type=str, default='gpt2',
                      help='Name of the GPT-2 model to use')
    parser.add_argument('--output-dir', type=str, default='artifacts/provenance',
                      help='Directory to save provenance and safety reports')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--min-age-rating', type=str, default='TEEN',
                      choices=['ALL_AGES', 'TEEN', 'MATURE', 'ADULT'],
                      help='Minimum age rating for content')
    parser.add_argument('--max-input-length', type=int, default=512,
                      help='Maximum input length')
    parser.add_argument('--max-output-length', type=int, default=100,
                      help='Maximum output length')
    parser.add_argument('--content-filters', type=str, nargs='+',
                      default=['bad_word', 'inappropriate', 'explicit'],
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
    
    # Initialize GPT-2 model
    model = GPT2Model(model_name=args.model_name)
    
    # Create training pipeline with safety checks
    pipeline = SafeTrainingPipeline(
        model=model,
        safety_config=safety_config,
        provenance_dir=str(output_dir)
    )
    
    # Load training data
    train_data = load_training_data()
    
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