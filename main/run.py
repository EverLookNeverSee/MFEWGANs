from torch import nn
from generator import Generator
from discriminator import Discriminator


# Setting learning rate, number of epochs and loss function
lr = 0.001
n_epochs = 300
loss_function = nn.BCELoss()
