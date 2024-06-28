import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, input_dim=25, output_dim=100, num_mlps=60, intermediate_output_dim=600, final_output_dim=368,
                 depth=6, final_depth=3):
        super(CustomModel, self).__init__()
        self.num_mlps = num_mlps
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the shared MLP with variable depth
        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
        self.shared_mlp = nn.Sequential(*layers)

        # Define the final projection layers with specified depth
        final_layers = []
        final_layers.append(nn.Linear(num_mlps * output_dim, intermediate_output_dim))
        final_layers.append(nn.ReLU())
        for _ in range(final_depth - 1):
            final_layers.append(nn.Linear(intermediate_output_dim, intermediate_output_dim))
            final_layers.append(nn.BatchNorm1d(intermediate_output_dim))
            final_layers.append(nn.ReLU())
        final_layers.append(nn.Linear(intermediate_output_dim, final_output_dim))
        self.final_projection = nn.Sequential(*final_layers)

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape to (N * 60, 25) to apply the MLP in parallel
        x_reshaped = x.view(-1, self.input_dim)  # Shape (N*60, 25)

        # Apply the shared MLP
        mlp_output = self.shared_mlp(x_reshaped)  # Shape (N*60, 25)

        # Reshape back to (N, 60 * 25)
        mlp_output_reshaped = mlp_output.view(batch_size, -1)  # Shape (N, 60*25)

        # Pass through the final projection layers
        final_output = self.final_projection(mlp_output_reshaped)  # Shape (N, final_output_dim)

        return final_output


