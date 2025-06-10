import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_data(batch_size=64, data_dir='data'):
    """
    Load MNIST dataset with transformations.
    
    Args:
        batch_size (int): Batch size for data loaders
        data_dir (str): Directory to store/load the data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        data_dir, 
        train=False, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader 