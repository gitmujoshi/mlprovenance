import gzip
import numpy as np
import torch
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 1, rows, cols) / 255.0
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def get_mnist_data():
    train_images = load_mnist_images(DATA_DIR / 'train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels(DATA_DIR / 'train-labels-idx1-ubyte.gz')
    test_images = load_mnist_images(DATA_DIR / 't10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels(DATA_DIR / 't10k-labels-idx1-ubyte.gz')

    # Convert to torch tensors
    train_images = torch.from_numpy(train_images)
    train_labels = torch.from_numpy(train_labels).long()
    test_images = torch.from_numpy(test_images)
    test_labels = torch.from_numpy(test_labels).long()

    # Return as TensorDataset for DataLoader compatibility
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    return train_dataset, test_dataset 