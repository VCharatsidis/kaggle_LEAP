import numpy as np
import torch
import math
import torch.nn as nn
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class SequenceToSequenceTransformer(nn.Module):
    def __init__(self, src_input_dim, tgt_input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, src_seq_length=60, tgt_seq_length=60):
        super(SequenceToSequenceTransformer, self).__init__()

        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.d_model = d_model
        self.output_dim = tgt_input_dim

        # Linear layers to project the input features to the model dimension
        self.src_input_linear = nn.Linear(src_input_dim, d_model)
        self.tgt_input_linear = nn.Linear(tgt_input_dim, d_model)
        self.output_linear = nn.Linear(d_model, tgt_input_dim)
        self.output_to_dmodel = nn.Linear(tgt_input_dim, d_model)

        # Positional encoding
        self.src_positional_encoding = PositionalEncoding(d_model, max_len=src_seq_length)
        self.tgt_positional_encoding = PositionalEncoding(d_model, max_len=tgt_seq_length)

        # Transformer Encoder to process the sequence
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.masks = {}
        for i in range(1, 61):
            self.masks[i] = self.generate_square_subsequent_mask(i).cuda().bool()

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # Encode the source sequence
        src = self.src_input_linear(src)
        src = self.src_positional_encoding(src)
        memory = self.transformer_encoder(src)

        # Initialize the output tensor
        tgt = self.tgt_input_linear(tgt)
        tgt = self.tgt_positional_encoding(tgt)

        outputs = []
        output = tgt[:, 0, :].unsqueeze(1)  # Start with the first token
        seq_len = tgt.size(1)
        teacher_forcing_decisions = np.random.rand(seq_len) < teacher_forcing_ratio

        for t in range(1, seq_len + 1):  # Loop from 1 to seq_len

            output_step = self.transformer_decoder(output,
                                                   memory,
                                                   tgt_mask=self.masks[t])

            next_token = self.output_linear(output_step[:, -1, :])
            outputs.append(next_token.unsqueeze(1))

            if t < seq_len:  # Only append next input if t < seq_len
                # Use the true target or the model's own prediction based on teacher forcing
                if teacher_forcing_decisions[t]:
                    next_input = tgt[:, t, :].unsqueeze(1)
                else:
                    next_input = self.output_to_dmodel(next_token).unsqueeze(1)

                output = torch.cat((output, next_input), dim=1)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def generate(self, src, max_len):
        # Encode the source sequence
        src = self.src_input_linear(src)
        src = self.src_positional_encoding(src)
        memory = self.transformer_encoder(src)

        # Start with the start-of-sequence token
        batch_size = src.size(0)
        tgt = torch.zeros(batch_size, 1, self.d_model).cuda()  # Initialize with zeros or start token in d_model space

        outputs = []

        for i in range(max_len):
            tgt_step = self.tgt_positional_encoding(tgt)
            output_step = self.transformer_decoder(tgt_step, memory, tgt_mask=self.masks[i + 1])
            next_token = self.output_linear(output_step[:, -1, :])
            outputs.append(next_token.unsqueeze(1))
            next_token_proj = self.output_to_dmodel(next_token).unsqueeze(1)
            tgt = torch.cat((tgt, next_token_proj), dim=1)

        return torch.cat(outputs, dim=1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        mask = mask.masked_fill(mask == 0, float(0.0))  # Fill the upper triangular part with -inf
        return mask

    def create_padding_mask(self, seq):
        return (seq == 0)  # Return a 2D mask

# Example usage
src_input_dim = 25
tgt_input_dim = 14
output_dim = 14
d_model = 32
nhead = 4
num_encoder_layers = 2
dim_feedforward = 64
dropout = 0.1
src_seq_length = 60
tgt_seq_length = 60

model = SequenceToSequenceTransformer(src_input_dim, tgt_input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, src_seq_length).cuda()

# Dummy input
src = torch.rand(16, src_seq_length, src_input_dim).cuda()  # batch_size=16, seq_length=60, src_input_dim=25
tgt = torch.rand(16, tgt_seq_length, tgt_input_dim).cuda()  # batch_size=16, seq_length=60, tgt_input_dim=14

# Define the teacher forcing ratio
teacher_forcing_ratio = 0.5  # Start with a high value and gradually decrease

output = model(src, tgt)

print(output.shape)  # Should print torch.Size([16, 60, 14])

max_len = 60
generated_output = model.generate(src, max_len)

print(generated_output.shape)  # Should print torch.Size([16, 60, 14])
