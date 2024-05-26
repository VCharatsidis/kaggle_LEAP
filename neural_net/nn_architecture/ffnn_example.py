import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN, self).__init__()

        # Initialize the layers
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # Normalization layer
            layers.append(nn.LeakyReLU(inplace=True))  # Activation
            layers.append(nn.Dropout(p=0.1))  # Dropout for regularization
            previous_size = hidden_size

        # Output layer - no dropout, no activation function
        layers.append(nn.Linear(previous_size, output_size))

        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FFNN_gelu(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN_gelu, self).__init__()

        # Initialize the layers
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, 2 * hidden_size))
            layers.append(nn.GLU(dim=1))  # Activation
            layers.append(nn.LayerNorm(hidden_size))  # Normalization layer

            #layers.append(nn.Dropout(p=0.1))  # Dropout for regularization
            previous_size = hidden_size

        # Output layer - no dropout, no activation function
        layers.append(nn.Linear(previous_size, output_size))

        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
