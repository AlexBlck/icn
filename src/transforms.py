import torch


class GaussianNoise(torch.nn.Module):

    def __init__(self, std=0.2):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std
