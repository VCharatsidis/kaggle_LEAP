import torch
import torch.nn as nn


class SoftFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.1):
        super(SoftFormer, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim * hidden_dim + hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)  # LayerNorm

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * hidden_dim + hidden_dim) for _ in range(num_layers)
        ])
        self.hidden_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)  # LayerNorm for each hidden layer
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def glu(self, x):

        values = x[:, :self.hidden_dim]
        gate = x[:, self.hidden_dim:]

        gate = gate.reshape(x.shape[0], self.hidden_dim, -1)

        gate = nn.functional.softmax(gate, dim=2)
        values = values.unsqueeze(2)
        result = values * gate
        result = result.mean(dim=1)

        # # Apply each linear layer to the corresponding slice of the tensor and remove the extra dimension
        # output_slices = [self.linear_layers_1[i](result[:, i, :]).squeeze(-1) for i in range(self.hidden_dim)]
        #
        # # Stack the results along the second dimension
        # output = torch.stack(output_slices, dim=1)  # Shape: [32, 128]

        return result

    def forward(self, x):
        x = self.glu(self.input_layer(x))
        x = self.input_norm(x)

        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            x = self.glu(layer(x))
            x = norm(x)

        x = self.output_layer(x)

        return x

#
# # Test script
# def test_softformer_forward_pass():
#     # Parameters
#     input_dim = 64
#     hidden_dim = 128
#     output_dim = 10
#     num_layers = 1
#     batch_size = 32
#
#     # Create a model instance
#     model = SoftFormer(input_dim, hidden_dim, output_dim, num_layers)
#
#     # Generate dummy input data
#     dummy_input = torch.randn(batch_size, input_dim)
#
#     # Forward pass
#     output = model(dummy_input)
#
#     # Check output shape
#     assert output.shape == (batch_size, output_dim), f"Expected output shape {(batch_size, output_dim)}, but got {output.shape}"
#
#     print("Forward pass test passed. Output shape:", output.shape)
#
# # Run the test
# test_softformer_forward_pass()