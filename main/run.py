import torch
from torch import nn, optim, ones, randn, zeros, cat
from generator import Generator
from discriminator import Discriminator
from prep import train_loader, batch_size


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


# Training process
for epoch in range(n_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = ones((batch_size, 1))
        latent_space_samples = randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = zeros((batch_size, 1))
        all_samples = cat((real_samples, generated_samples))
        all_samples_labels = cat((real_samples_labels, generated_samples_labels))

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = randn((batch_size, 2))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss value
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} , Loss D: {loss_discriminator}")
            print(f"Epoch: {epoch} , Loss G: {loss_generator}")
