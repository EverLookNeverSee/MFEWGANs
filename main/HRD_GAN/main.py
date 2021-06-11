"""
    Generating hand-written digits using GANs
"""

import math
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from classes import Generator, Discriminator


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

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

# Plotting some of samples
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
plt.show()

# Instantiating
generator = Generator().to(device=device)
discriminator = Discriminator().to(device=device)

# Setting training parameters
lr = 0.001
n_epochs = 50
loss_function = nn.BCELoss()

# Configuring optimizer
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(n_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
        latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
