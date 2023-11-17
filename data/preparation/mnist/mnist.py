import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def prepare_loaders(batch_size):

    train_data = datasets.MNIST(
        root='../../data/datasets',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    test_data = datasets.MNIST(
        root='../../data/datasets',
        train=False,
        transform=ToTensor(),
        download=True,
    )

    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1),

        'test': torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=1),
    }

    return loaders
