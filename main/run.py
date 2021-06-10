from torch import nn, optim
from generator import Generator
from discriminator import Discriminator


# Setting learning rate, number of epochs and loss function
lr = 0.001
n_epochs = 300
loss_function = nn.BCELoss()

# Instantiating
generator = Generator()
discriminator = Discriminator()


# Setting optimization algorithm
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = optim.Adam(generator.parameters(), lr=lr)
