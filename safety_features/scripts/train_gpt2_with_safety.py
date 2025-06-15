import argparse
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import time
import datetime
import os
import requests
from tqdm import tqdm
import json
import platform
import psutil
import hashlib

from safety_features.safety_checks import SafetyConfig, AgeRating
from safety_features.train_with_safety import SafeTrainingPipeline
from safety_features.safety_checks import SafetyChecker

# Configure logging with timestamp and more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GPT2Model:
    def __init__(self, model_name="gpt2", safety_config=None):
        logger.info(f"Initializing GPT2Model with {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create cache directory if it doesn't exist
        cache_dir = Path.home() / ".cache" / "huggingface" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=False,
                resume_download=True
            )
            
            logger.info("Loading model...")
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=False,
                resume_download=True
            ).to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading model/tokenizer: {str(e)}")
            logger.info("Attempting to download with alternative method...")
            
            # Try alternative download method
            try:
                self._download_model_alternative(model_name, cache_dir)
            except Exception as e2:
                logger.error(f"Alternative download also failed: {str(e2)}")
                raise RuntimeError("Failed to load model after multiple attempts")
        
        logger.info("Initializing safety checker...")
        self.safety_checker = SafetyChecker(safety_config) if safety_config else None
        if self.safety_checker:
            logger.info(f"Safety checker initialized with config: {safety_config.to_dict()}")
        
        self.model.train()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _download_model_alternative(self, model_name: str, cache_dir: Path):
        """Alternative method to download model files."""
        base_url = f"https://huggingface.co/{model_name}/resolve/main"
        files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"
        ]
        
        model_dir = cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            url = f"{base_url}/{file}"
            output_path = model_dir / file
            
            if not output_path.exists():
                logger.info(f"Downloading {file}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(output_path, 'wb') as f, tqdm(
                    desc=file,
                    total=total_size,
                    unit='iB',
                    unit_scale=True
                ) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)
        
        # Now try loading from local files
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
        self.model = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(self.device)

    def generate(self, input_text: str, **kwargs):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def update(self, texts):
        """
        Update the model with a batch of texts.
        Returns the loss value.
        """
        if not texts:
            logger.warning("Empty batch received, skipping update")
            return 0.0
            
        # Apply safety checks if configured
        if self.safety_checker:
            logger.info(f"Applying safety checks to batch of {len(texts)} texts")
            safe_texts = self.safety_checker.check_batch(texts)
            if not safe_texts:
                logger.warning("No safe texts in batch, skipping update")
                return 0.0
            texts = safe_texts
            
        # Prepare batch
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Forward pass
        outputs = self.model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        return loss.item()

def load_training_data(batch_size, sample_size=None):
    logger.info("Loading wikitext dataset...")
    try:
        # Try loading with cache
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=str(Path.home() / ".cache" / "huggingface" / "datasets"),
            download_mode="force_redownload"
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.info("Attempting to download dataset with alternative method...")
        
        # Create dataset directory
        dataset_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset files
        base_url = "https://huggingface.co/datasets/wikitext/resolve/main/data/wikitext-2-raw-v1"
        files = ["train.txt", "validation.txt", "test.txt"]
        
        for file in files:
            url = f"{base_url}/{file}"
            output_path = dataset_dir / file
            
            if not output_path.exists():
                logger.info(f"Downloading {file}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(output_path, 'wb') as f, tqdm(
                    desc=file,
                    total=total_size,
                    unit='iB',
                    unit_scale=True
                ) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)
        
        # Try loading from local files
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=str(dataset_dir),
            local_files_only=True
        )
    
    # Apply sample size if specified
    if sample_size is not None:
        logger.info(f"Using sample size of {sample_size} examples")
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    logger.info("Preprocessing dataset...")
    def preprocess_function(examples):
        filtered = [text for text in examples["text"] if text and text.strip()]
        return {"text": filtered}
    
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    processed_dataset = processed_dataset.filter(lambda example: example["text"] and example["text"].strip())
    logger.info(f"Preprocessing complete. {len(processed_dataset)} examples remaining")
    
    def collate_fn(batch):
        return [item["text"] for item in batch if item["text"] and item["text"].strip()]
    
    dataloader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    logger.info(f"DataLoader created with batch size {batch_size}")
    return dataloader

class ProvenanceTracker:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "training_config": {},
            "data_provenance": {},
            "model_provenance": {},
            "safety_metrics": {},
            "epoch_metrics": []
        }

    def _get_system_info(self):
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

    def update_training_config(self, config):
        self.provenance_data["training_config"] = config

    def update_data_provenance(self, dataset_info):
        self.provenance_data["data_provenance"] = dataset_info

    def update_model_provenance(self, model_info):
        self.provenance_data["model_provenance"] = model_info

    def update_safety_metrics(self, metrics):
        self.provenance_data["safety_metrics"] = metrics

    def add_epoch_metrics(self, epoch, metrics):
        self.provenance_data["epoch_metrics"].append({
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def save_provenance_report(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"provenance_report_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(self.provenance_data, f, indent=2)
        return output_file

class MerkleTree:
    def __init__(self, data):
        self.data = data
        section_nodes = []
        for section_name in ['timestamp', 'system_info', 'training_config', 'data_provenance', 'model_provenance', 'safety_metrics', 'epoch_metrics']:
            if section_name in data:
                child = self._build_tree(data[section_name], name=section_name)
                if child is not None:
                    if isinstance(child, dict) and child.get('name') == section_name:
                        section_nodes.append(child)
                    else:
                        section_nodes.append({
                            'hash': child['hash'] if isinstance(child, dict) and 'hash' in child else self._hash(child),
                            'name': section_name,
                            'child': child
                        })
        # Instead of building a tree from pairs, keep each section as a separate node
        self.tree = section_nodes

    def _hash(self, item):
        if isinstance(item, dict):
            item = json.dumps(item, sort_keys=True)
        elif isinstance(item, (list, tuple)):
            item = json.dumps(item, sort_keys=True)
        return hashlib.sha256(str(item).encode()).hexdigest()

    def _build_tree(self, data, name=None):
        if not data:
            return None
        if isinstance(data, dict):
            leaves = [self._build_tree(v, k) for k, v in sorted(data.items())]
        elif isinstance(data, (list, tuple)):
            leaves = [self._build_tree(item, f"{name}[{i}]" if name else str(i)) for i, item in enumerate(data)]
        else:
            return {"hash": self._hash(data), "name": name}
        
        if len(leaves) == 1:
            return leaves[0]
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        pairs = [leaves[i:i+2] for i in range(0, len(leaves), 2)]
        return [
            {
                "hash": self._hash(pair),
                "name": " + ".join([leaf.get("name", "") for leaf in pair if isinstance(leaf, dict) and leaf.get("name")])
            }
            for pair in pairs
        ]

    def get_root(self):
        if not self.tree:
            return None
        # Compute the root hash from all section nodes
        return self._hash([node['hash'] for node in self.tree])

    def to_dict(self):
        return {
            "root": self.get_root(),
            "tree": self.tree
        }

def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 with safety features')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sample-size', type=int, default=None, help='Number of examples to use for testing (default: use full dataset)')
    parser.add_argument('--min-age-rating', type=str, default='TEEN')
    parser.add_argument('--max-input-length', type=int, default=512)
    parser.add_argument('--content-filters', nargs='+', default=['violence', 'explicit', 'offensive'])
    parser.add_argument('--block-sensitive-topics', action='store_true')
    parser.add_argument('--require-content-warning', action='store_true')
    parser.add_argument('--output-dir', type=str, default='artifacts/provenance', help='Directory to save provenance reports')
    args = parser.parse_args()

    # Create a unique run folder using timestamp
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(args.output_dir) / f"run_{run_timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run artifacts will be saved in: {run_folder}")

    # Initialize provenance tracker with the run folder
    provenance_tracker = ProvenanceTracker(run_folder)

    # Create safety configuration
    safety_config = SafetyConfig(
        min_age_rating=args.min_age_rating,
        max_input_length=args.max_input_length,
        content_filters=args.content_filters,
        block_sensitive_topics=args.block_sensitive_topics,
        require_content_warning=args.require_content_warning
    )

    # Update training config in provenance
    provenance_tracker.update_training_config({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sample_size": args.sample_size,
        "safety_config": safety_config.to_dict()
    })

    # Initialize model with safety config
    model = GPT2Model(safety_config=safety_config)
    
    # Update model provenance
    provenance_tracker.update_model_provenance({
        "model_name": "gpt2",
        "model_type": "GPT2LMHeadModel",
        "device": str(model.device),
        "safety_config": safety_config.to_dict()
    })
    
    # Load training data
    train_loader = load_training_data(batch_size=args.batch_size, sample_size=args.sample_size)
    
    # Update data provenance
    provenance_tracker.update_data_provenance({
        "dataset_name": "wikitext-2-raw-v1",
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "total_examples": len(train_loader.dataset)
    })
    
    # Training loop
    start_time = time.time()
    logger.info(f"Starting training with configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Safety config: {safety_config.to_dict()}")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_losses = []
        epoch_safety_metrics = {
            "total_checks": 0,
            "passed_checks": 0,
            "content_warnings": 0,
            "filter_violations": {}
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Process batch with safety checks
            safe_batch = model.safety_checker.check_batch(batch)
            
            # Update safety metrics
            epoch_safety_metrics["total_checks"] += len(batch)
            epoch_safety_metrics["passed_checks"] += len(safe_batch)
            
            if not safe_batch:
                logger.warning(f"Batch {batch_idx}: No safe texts found, skipping")
                continue
                
            # Update model with safe batch
            loss = model.update(safe_batch)
            epoch_losses.append(loss)
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses[-10:]))
                logger.info(f"Batch {batch_idx + 1}: Average loss = {avg_loss:.4f}")
        
        # Calculate epoch metrics
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        logger.info(f"Epoch {epoch + 1} complete:")
        logger.info(f"  Average loss: {epoch_avg_loss:.4f}")
        logger.info(f"  Total batches processed: {len(epoch_losses)}")
        
        # Update epoch metrics in provenance
        epoch_metrics = {
            "epoch": epoch + 1,
            "average_loss": epoch_avg_loss,
            "total_batches": len(epoch_losses),
            "safety_metrics": epoch_safety_metrics
        }
        provenance_tracker.add_epoch_metrics(epoch + 1, epoch_metrics)
    
    # Update final safety metrics
    provenance_tracker.update_safety_metrics(model.safety_checker.metrics.__dict__)
    
    # Save provenance report
    output_file = provenance_tracker.save_provenance_report()
    logger.info(f"\nProvenance report saved to: {output_file}")
    
    # Compute Merkle tree from provenance data and dump it
    merkle_tree = MerkleTree(provenance_tracker.provenance_data)
    merkle_output_file = run_folder / f"merkle_tree_{run_timestamp}.json"
    with open(merkle_output_file, 'w') as f:
        json.dump(merkle_tree.to_dict(), f, indent=2)
    logger.info(f"Merkle tree dumped to: {merkle_output_file}")
    
    # Save the trained model
    model_save_path = run_folder / "trained_model.pt"
    torch.save(model.model.state_dict(), model_save_path)
    logger.info(f"Trained model saved to: {model_save_path}")
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time:.2f} seconds")
    logger.info(f"Final safety metrics:")
    logger.info(f"  Total checks: {model.safety_checker.metrics.total_checks}")
    logger.info(f"  Pass rate: {model.safety_checker.metrics.passed_checks/model.safety_checker.metrics.total_checks:.2%}")
    logger.info(f"  Content warnings: {model.safety_checker.metrics.content_warnings}")
    logger.info(f"  Most common violations: {model.safety_checker.metrics.filter_violations}")

if __name__ == "__main__":
    main() 