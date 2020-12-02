import torch


class GaussianNoise:

    def __init__(self, std=0.2):
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std
