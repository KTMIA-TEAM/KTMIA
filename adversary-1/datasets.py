import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import TensorDataset, ConcatDataset


def get_dataset(name, train=True, augment=False):
    print(f"Build Dataset {name}")
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = torchvision.datasets.CIFAR10(root='data/datasets/cifar10-data', train=train, download=True, 
                                               transform=transform)

    elif name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = torchvision.datasets.CIFAR100(root='data/datasets/cifar100-data', train=train, download=True, 
                                                transform=transform)

    elif name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        dataset = torchvision.datasets.MNIST(root='data/datasets/mnist-data', train=train, download=True,
                                             transform=transform)

    elif name == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = torchvision.datasets.SVHN(root='data/datasets/svhn-data', split='train' if train else "test", download=True, 
                                            transform=transform)

    elif name == "texas100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz
        dataset = np.load("data/datasets/texas/data_complete.npz")
        x_data = torch.tensor(dataset['x'][:, :]).float()
        y_data = torch.tensor(dataset['y'][:] - 1).long()
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            dataset = None

    elif name == "location":
        # the dataset can be downloaded from https://github.com/jjy1994/MemGuard/tree/master/data/location
        dataset = np.load("data/datasets/location/data_complete.npz")
        x_data = torch.tensor(dataset['x'][:, :]).float()
        y_data = torch.tensor(dataset['y'][:] - 1).long()
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            dataset = None

    elif name == "purchase100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz
        tensor_path = "data/datasets/purchase100/purchase100.npz"
        if os.path.exists(tensor_path):
            data = np.load(tensor_path)
            x_data = torch.tensor(data['features']).float()
            y_data = torch.nonzero(torch.tensor(data['labels']))[:,1].long()
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            dataset = None
    
    elif name == "news":
        data = np.load("data/datasets/20news/20news.npz")
        x_data = torch.tensor(data['features']).float()
        y_data = torch.tensor(data['labels']).long()
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            dataset = None

    elif name == "cinic":
        cinic_directory = "data/datasets/cinic"
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if train:
            dataset = torchvision.datasets.ImageFolder(cinic_directory+'/train', transform=transform)
        else:
            valid_dataset = torchvision.datasets.ImageFolder(cinic_directory+'/valid', transform=transform)
            test_dataset = torchvision.datasets.ImageFolder(cinic_directory+'/test', transform=transform)
            dataset = ConcatDataset([valid_dataset, test_dataset])
    else:
        raise ValueError

    return dataset