import torch
from imagenet_c import corrupt, corruption_dict
import random
import numpy as np


class ImagenetC(torch.nn.Module):
    methods = list(corruption_dict.keys())

    def __init__(self, max_sev):
        super().__init__()
        self.max_sev = max_sev

    def forward(self, x):
        method = random.choice(self.methods)
        severity = random.randint(0, self.max_sev)

        x = np.array(x)
        if random.random() > 0.5:
            x = corrupt(x, severity, method)
        return x
