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

# mnist_train = dset.MNIST(root="datasets/mnist",
#                          train=True,
#                          download=True,
#                          transform=input_transforms)
# mnist_test = dset.MNIST(root="datasets/mnist",
#                         train=False,
#                         download=True,
#                         transform=input_transforms)

cifar_train = dset.CIFAR10(root="datasets/cifar",
                           train=True,
                           transform=input_transforms,
                           download=True)

cifar_test = dset.CIFAR10(root="datasets/cifar",
                          train=False,
                          transform=input_transforms,
                          download=True)
train_set = cifar_train
test_set = cifar_test

indices = list(range(len(train_set)))
train_indices = indices[:int(len(train_set) * 0.9)]
val_indices = indices[int(len(train_set) * 0.9):]

dataloaders = {"train": DataLoader(Subset(train_set, train_indices), shuffle=True, batch_size=128),
               "val": DataLoader(Subset(train_set, val_indices), shuffle=False, batch_size=128),
               "test": DataLoader(test_set, shuffle=False, batch_size=128)}

# model = models.SimpleMLP()
# model = models.SimpleConv()
model = models.VGG(train_set[0][0].size(0), len(train_set.classes))
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)
train.train_mnist(device, model, dataloaders, 30)
