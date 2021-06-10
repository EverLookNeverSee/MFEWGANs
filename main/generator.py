"""
    This file contains the generator class
"""

from torch import nn


class Generator(nn.Module):
    """ Implementation of generator """
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
