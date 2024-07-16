import torch
import torch.nn as nn


class GLU_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.1):
        super(GLU_MLP, self).__init__()
        self.num_layers = num_layers

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim * 2)
        self.input_norm = nn.LayerNorm(hidden_dim)  # LayerNorm
        self.input_dropout = nn.Dropout(dropout_prob)  # Dropout for the input layer

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(num_layers)
        ])
        self.hidden_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)  # LayerNorm for each hidden layer
        ])
        self.hidden_dropouts = nn.ModuleList([
            nn.Dropout(dropout_prob) for _ in range(num_layers)  # Dropout for each hidden layer
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_dropout = nn.Dropout(dropout_prob)  # Dropout for the output layer

    def glu(self, x):
        x, gate = x.chunk(2, dim=-1)
        gate = nn.functional.softmax(gate, dim=-1)
        return x * gate #+ x

    def forward(self, x):
        x = self.glu(self.input_layer(x))
        x = self.input_norm(x)
        # x = self.input_dropout(x)  # Apply dropout after LayerNorm in the input layer

        for layer, norm, dropout in zip(self.hidden_layers, self.hidden_norms, self.hidden_dropouts):
            x = self.glu(layer(x))
            x = norm(x)
            # x = dropout(x)  # Apply dropout after each LayerNorm in hidden layers

        x = self.output_layer(x)

        return x
