
import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ModifiedSequenceToScalarTransformer_positional(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, seq_length=60):
        super(ModifiedSequenceToScalarTransformer_positional, self).__init__()

        self.seq_length = seq_length

        # Linear layer to project the input features to the model dimension
        self.input_linear = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_length)

        # Transformer Encoder to process the sequence
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        # Output linear layer that maps from the flattened model dimension to the desired output dimension
        self.output_linear = nn.Linear(d_model * seq_length, output_dim)

    def forward(self, src):
        src = self.input_linear(src)  # shape: [batch_size, seq_len, d_model]

        # Add positional encoding
        src = self.positional_encoding(src)  # shape: [batch_size, seq_len, d_model]

        # Process sequence with the transformer encoder
        src = self.transformer_encoder(src)  # shape: [batch_size, seq_len, d_model]

        # Flatten the sequence output
        src = src.view(src.size(0), -1)  # shape: [batch_size, seq_len * d_model]

        # Map to the desired output dimension
        output = self.output_linear(src)  # shape: [batch_size, output_dim]

        return output