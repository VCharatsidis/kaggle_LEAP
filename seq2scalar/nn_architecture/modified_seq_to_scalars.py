import torch.nn as nn


class ModifiedSequenceToScalarTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, seq_length=60):
        super(ModifiedSequenceToScalarTransformer, self).__init__()

        self.seq_length = seq_length

        # Linear layer to project the input features to the model dimension
        self.input_linear = nn.Linear(input_dim, d_model)

        # Transformer Encoder to process the sequence
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(batch_first=True, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )

        # Output linear layer that maps from the flattened model dimension to the desired output dimension
        self.output_linear = nn.Linear(d_model * seq_length, output_dim)

    def forward(self, src):
        src = self.input_linear(src)  # shape: [batch_size, seq_len, d_model]

        # Process sequence with the transformer encoder
        src = self.transformer_encoder(src)  # shape: [batch_size, seq_len, d_model]

        # Flatten the sequence output
        src = src.view(src.size(0), -1)  # shape: [batch_size, seq_len * d_model]

        # Map to the desired output dimension
        output = self.output_linear(src)  # shape: [batch_size, output_dim]

        return output
