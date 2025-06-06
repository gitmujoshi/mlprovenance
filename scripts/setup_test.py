import sys
import torch
import opacus
import numpy as np
import sklearn
import tqdm
import psutil
import git
import pytest

# Test the new MNIST loader
from pathlib import Path
import gzip
import urllib.request

MNIST_MIRRORS = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
]

def test_imports():
    print("Testing imports...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Opacus version: {opacus.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"tqdm version: {tqdm.__version__}")
    print(f"psutil version: {psutil.__version__}")
    print(f"gitpython version: {git.__version__}")
    print(f"pytest version: {pytest.__version__}")

def test_cuda():
    print("\nTesting CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

def test_opacus():
    print("\nTesting Opacus setup...")
    from opacus import PrivacyEngine
    print("Opacus PrivacyEngine imported successfully")

def test_mnist_data():
    print("\nTesting MNIST data loading (direct download with fallback)...")
    DATA_DIR = Path('data')
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / 'train-images-idx3-ubyte.gz'
    if not out_path.exists():
        for url in MNIST_MIRRORS:
            print(f"Trying {url}...")
            try:
                urllib.request.urlretrieve(url, out_path)
                print(f"Downloaded from {url}")
                break
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
        else:
            print("ERROR: Could not download MNIST data from any known mirror.")
            return
    try:
        with gzip.open(out_path, 'rb') as f:
            f.read(4)
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            buf = f.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, rows, cols) / 255.0
            print(f"MNIST training images loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading MNIST data: {str(e)}")

if __name__ == "__main__":
    print("Starting setup test...\n")
    test_imports()
    test_cuda()
    test_opacus()
    test_mnist_data()
    print("\nSetup test completed!") 