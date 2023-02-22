import torch
import torch.nn as nn

class ActivationFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}

class Sigmoid(ActivationFunction):

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

class Tanh(ActivationFunction):
    
    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

class ReLU(ActivationFunction):

    def forward(self, x):
        return x * (x > 0).float()

class LeakyReLU(ActivationFunction):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.config["alpha"] = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config["alpha"] * x)

class ELU(ActivationFunction):
    
    def forward(self, x):
        return torch.where(x > 0, x, torch.exp(x) - 1)

class Swish(ActivationFunction):

    def forward(self, x):
        return x * torch.sigmoid(x)