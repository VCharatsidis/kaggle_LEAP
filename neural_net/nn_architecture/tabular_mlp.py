import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularModel(nn.Module):
    def __init__(self, dims: list):
        """
        Initializes the LeapModel.

        Parameters
        ----------
        dims : list of int
            A list containing the dimensions of each layer in the network.
            The length of the list determines the number of layers.
        """

        super().__init__()

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x)
        return y

