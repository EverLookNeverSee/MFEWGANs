"""
    Generating hand-written digits using GANs
"""

import math
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# Setting up seed
torch.manual_seed(111)

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Preparing the training data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)
