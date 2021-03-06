"""
    This file contains preprocessing stuff
"""

import math
import torch
import matplotlib.pyplot as plt

# Setting seed parameter
torch.manual_seed(111)

# Training data preparation
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

# Creating data loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

if __name__ == '__main__':
    plt.plot(train_data[:, 0], train_data[:, 1], ".")
    plt.show()
