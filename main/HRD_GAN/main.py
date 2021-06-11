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
