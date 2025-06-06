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

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.provenance.tracker import ProvenanceTracker
from src.provenance.verifier import ProvenanceVerifier

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

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_model():
    """Create the MNIST model."""
    model = MNISTModel()
    return model

def train_model():
    """Train the MNIST model with differential privacy and provenance tracking."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
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
    
    # Training configuration
    config = {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "privacy_parameters": {
            "target_epsilon": 1.0,
            "target_delta": 1e-5,
            "max_grad_norm": 1.0
        }
    }
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Initialize privacy engine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.0,
        max_grad_norm=config["privacy_parameters"]["max_grad_norm"],
    )
    
    # Training loop
    logger.info("Training model...")
    training_logs = []
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
            # Get privacy metrics
            if batch_idx % 100 == 0:
                epsilon = privacy_engine.get_epsilon(config["privacy_parameters"]["target_delta"])
                logger.info(f"Privacy budget used: {epsilon:.2f}")
        
        # Calculate epoch metrics
        train_accuracy = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        test_accuracy = test_correct / test_total
        test_loss = test_loss / len(test_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}:")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        training_logs.append({
            "epoch": epoch + 1,
            "accuracy": train_accuracy,
            "loss": train_loss,
            "val_accuracy": test_accuracy,
            "val_loss": test_loss
        })
    
    # Get final privacy metrics
    final_epsilon = privacy_engine.get_epsilon(config["privacy_parameters"]["target_delta"])
    privacy_budget_used = (final_epsilon / config["privacy_parameters"]["target_epsilon"]) * 100
    
    # Calculate privacy impact on performance
    baseline_accuracy = 0.95  # Typical MNIST accuracy without privacy
    performance_impact = f"{((test_accuracy - baseline_accuracy) / baseline_accuracy) * 100:.2f}%"
    
    # Track privacy metrics
    privacy_metrics = {
        "achieved_epsilon": final_epsilon,
        "achieved_delta": config["privacy_parameters"]["target_delta"],
        "privacy_budget_used": privacy_budget_used,
        "secure_rng_enabled": False,  # We're in experimental mode
        "performance_impact": performance_impact,
        "privacy_utility_tradeoff": "Balanced" if privacy_budget_used < 80 else "Privacy-focused"
    }
    
    # Track training provenance with privacy metrics
    provenance.track_training(
        model=model,
        config=config,
        training_logs=training_logs,
        final_metrics={
            "final_accuracy": test_accuracy,
            "final_loss": test_loss
        },
        privacy_metrics=privacy_metrics
    )
    
    # Save model
    model_dir = project_root / "artifacts" / "models" / provenance.timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    
    # Save provenance data
    provenance.save()
    
    # Verify training
    logger.info("Verifying training...")
    verifier = ProvenanceVerifier(provenance.provenance_dir)
    verification_report = verifier.generate_verification_report(model_dir / "model.pth")
    
    # Generate Merkle proofs for all components
    merkle_proofs = {}
    for comp in ["data", "model", "training"]:
        comp_data = None
        if comp == "data":
            comp_data = provenance.data["data_provenance"]
        elif comp == "model":
            comp_data = provenance.data["model_provenance"]
        elif comp == "training":
            comp_data = provenance.data["training_provenance"]
        merkle_proofs[comp] = provenance.get_provenance_proof(comp, comp_data)
    
    # Generate final report
    from src.provenance.generate_final_report import generate_final_report
    report_path = generate_final_report(
        provenance.provenance_dir,
        model_dir / "model.pth",
        verification_report=verification_report,
        merkle_proofs=merkle_proofs
    )
    logger.info(f"Final report generated at {report_path}")
    
    return model, provenance

if __name__ == "__main__":
    train_model() 