import os
import urllib.request
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
import sys
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Tuple, Dict, Any
import torch.nn.functional as F

# Update imports to use package imports
from ml_provenance.models.mnist_model import MNISTModel
from ml_provenance.data.mnist_data import get_mnist_data
from ml_provenance.provenance.tracker import Tracker, ProvenanceTracker
from ml_provenance.provenance.verifier import Verifier, ProvenanceVerifier
from ml_provenance.provenance.report_generator import ReportGenerator
from ml_provenance.provenance.generate_final_report import generate_final_report

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# URLs for MNIST data
MNIST_MIRRORS = {
    'train_images': [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz'
    ],
    'train_labels': [
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz'
    ],
    'test_images': [
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz'
    ],
    'test_labels': [
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'
    ]
}

DATA_DIR = project_root / 'data'


def download_mnist():
    DATA_DIR.mkdir(exist_ok=True)
    for key, urls in MNIST_MIRRORS.items():
        out_path = DATA_DIR / urls[0].split('/')[-1]
        if not out_path.exists():
            print(f"Downloading {key}...")
            for url in urls:
                try:
                    print(f"Trying {url}...")
                    urllib.request.urlretrieve(url, out_path)
                    print(f"Successfully downloaded from {url}")
                    break
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
            else:
                raise Exception(f"Could not download {key} from any mirror")


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows, cols) / 255.0
        return data


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def get_mnist_datasets():
    download_mnist()
    train_images = load_mnist_images(DATA_DIR / 'train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels(DATA_DIR / 'train-labels-idx1-ubyte.gz')
    test_images = load_mnist_images(DATA_DIR / 't10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels(DATA_DIR / 't10k-labels-idx1-ubyte.gz')
    return train_images, train_labels, test_images, test_labels

def create_model():
    """Create the MNIST model."""
    model = MNISTModel()
    return model

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    privacy_engine: PrivacyEngine,
    epochs: int,
    device: str,
    provenance_tracker: ProvenanceTracker,
    target_delta: float,
    noise_multiplier: float,
    config: dict
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the model with differential privacy."""
    model = model.to(device)
    model.train()
    
    # Initialize tracking
    training_history = []
    best_accuracy = 0.0
    last_epoch_loss = 0.0
    last_epoch_train_acc = 0.0
    last_epoch_test_acc = 0.0
    privacy_budget = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total  # fraction
        
        # Test the model
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_accuracy = test_correct / test_total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # Get privacy metrics from the privacy engine
        current_epsilon = privacy_engine.get_epsilon(target_delta)
        privacy_budget.append(current_epsilon)
        privacy_metrics = {
            'epsilon': current_epsilon,
            'delta': target_delta,
            'noise_multiplier': noise_multiplier,
            'privacy_budget': privacy_budget
        }
        
        # Track epoch metrics and model state
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': epoch_loss,  # Using same loss for now
            'train_acc': epoch_accuracy,
            'val_acc': test_accuracy,
            'model_state': model.state_dict(),  # Capture model state
            'is_final': epoch == epochs - 1,
            'privacy_metrics': privacy_metrics
        }
        
        # Add to training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': epoch_loss,
            'train_acc': epoch_accuracy,
            'val_acc': test_accuracy
        })
        
        # Update provenance with epoch data
        provenance_tracker.update_training_provenance(epoch_data)
        
        last_epoch_loss = epoch_loss
        last_epoch_train_acc = epoch_accuracy
        last_epoch_test_acc = test_accuracy
    
    final_metrics = {
        'loss': last_epoch_loss,
        'train_accuracy': last_epoch_train_acc,
        'test_accuracy': last_epoch_test_acc,
        'training_history': training_history,
        'privacy_metrics': {
            'target_epsilon': config['privacy_parameters']['target_epsilon'],
            'final_epsilon': privacy_budget[-1] if privacy_budget else 0,
            'delta': target_delta,
            'privacy_budget': privacy_budget
        }
    }
    
    return model, final_metrics

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Training configuration
    config = {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "hash_algorithm": "sha256",  # Can be: sha256, blake3, sha512, sha1, md5
        "privacy_parameters": {
            "target_epsilon": 1.0,
            "target_delta": 1e-5,
            "max_grad_norm": 1.0,
            "noise_multiplier": 1.0
        }
    }
    
    # Configure hash function from training config
    from ml_provenance.provenance.hash_config import TrainingHashConfig
    hash_config = TrainingHashConfig(config)
    hash_config.initialize_hash_factory()
    
    logger.info(f"Using hash algorithm: {hash_config.get_current_algorithm()}")
    logger.info(f"Algorithm info: {hash_config.get_algorithm_info()}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = get_mnist_datasets()
    logger.info(f"Training data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    
    # Convert to torch tensors
    x_train_tensor = torch.from_numpy(x_train).unsqueeze(1)  # (N, 1, 28, 28)
    y_train_tensor = torch.from_numpy(y_train).long()
    x_test_tensor = torch.from_numpy(x_test).unsqueeze(1)
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # Initialize provenance tracker
    provenance = ProvenanceTracker()
    
    # Track data provenance
    provenance.track_data(x_train, x_test)
    
    # Create model
    logger.info("Creating model...")
    model = create_model().to(device)
    
    # Track model provenance
    provenance.track_model(model)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Initialize privacy engine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=config["privacy_parameters"]["noise_multiplier"],
        max_grad_norm=config["privacy_parameters"]["max_grad_norm"],
    )
    
    # Train the model
    model, final_metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        epochs=config["epochs"],
        device=device,
        provenance_tracker=provenance,
        target_delta=config["privacy_parameters"]["target_delta"],
        noise_multiplier=config["privacy_parameters"]["noise_multiplier"],
        config=config
    )
    
    # Save the model
    model_dir = project_root / "artifacts" / "models" / provenance.timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    
    # Save provenance data
    provenance.set_final_metrics(final_metrics, config)
    provenance.save()
    
    # Verify provenance
    verifier = ProvenanceVerifier(provenance.provenance_dir)
    verification_report = verifier.generate_verification_report(model_path=model_dir / "model.pth")
    
    # Generate final report
    generate_final_report(
        str(model_dir / "model.pth"),
        str(provenance.provenance_dir),
        config
    )

if __name__ == "__main__":
    main() 