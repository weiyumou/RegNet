import random

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import models
import train

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_transforms = transforms.Compose([transforms.ToTensor()])

mnist_train = dset.MNIST(root="datasets/mnist",
                         train=True,
                         download=True,
                         transform=input_transforms)
mnist_test = dset.MNIST(root="datasets/mnist",
                        train=False,
                        download=True,
                        transform=input_transforms)

indices = list(range(len(mnist_train)))
train_indices = indices[:int(len(mnist_train) * 0.9)]
val_indices = indices[int(len(mnist_train) * 0.9):]

dataloaders = {"train": DataLoader(Subset(mnist_train, train_indices), shuffle=True, batch_size=128),
               "val": DataLoader(Subset(mnist_train, val_indices), shuffle=False, batch_size=128),
               "test": DataLoader(mnist_test, shuffle=False, batch_size=128)}

model = models.SimpleMLP()
train.train_mnist(device, model, dataloaders, 10)
