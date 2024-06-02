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
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

    def forward(self, src, tgt):
        # Encode the source sequence
        src = self.src_input_linear(src)
        src = self.src_positional_encoding(src)
        memory = self.transformer_encoder(src)

        # Initialize the output tensor
        tgt = self.tgt_input_linear(tgt)
        tgt = self.tgt_positional_encoding(tgt)

        # Decode the target sequence
        output = self.transformer_decoder(tgt, memory)

        # Map to the desired output dimension
        output = self.output_linear(output)

        return output

    def generate(self, src, max_len):
        # Encode the source sequence
        src = self.src_input_linear(src)
        src = self.src_positional_encoding(src)
        memory = self.transformer_encoder(src)

        # Start with the start-of-sequence token
        batch_size = src.size(0)
        tgt = torch.zeros(batch_size, 1, self.d_model).cuda()  # Initialize with zeros or start token in d_model space

        outputs = []

        for _ in range(max_len):
            tgt_step = self.tgt_positional_encoding(tgt)
            output_step = self.transformer_decoder(tgt_step, memory)
            next_token = self.output_linear(output_step[:, -1, :])
            outputs.append(next_token.unsqueeze(1))
            next_token_proj = self.output_to_dmodel(next_token).unsqueeze(1)
            tgt = torch.cat((tgt, next_token_proj), dim=1)

        return torch.cat(outputs, dim=1)


# # Example usage
# src_input_dim = 25
# tgt_input_dim = 14
# output_dim = 14
# d_model = 32
# nhead = 4
# num_encoder_layers = 2
# dim_feedforward = 64
# dropout = 0.1
# src_seq_length = 60
# tgt_seq_length = 60
#
# model = SequenceToSequenceTransformer(src_input_dim, tgt_input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, src_seq_length).cuda()
#
# # Dummy input
# src = torch.rand(16, src_seq_length, src_input_dim).cuda()  # batch_size=16, seq_length=60, src_input_dim=558
# tgt = torch.rand(16, tgt_seq_length, tgt_input_dim).cuda()  # batch_size=16, seq_length=60, tgt_input_dim=368
# output = model(src, tgt)
#
# print(output.shape)  # Should print torch.Size([16, 60, 368])
#
# max_len = 60
# generated_output = model.generate(src, max_len)
#
# print(generated_output.shape)  # Should print torch.Size([16, 60, 368])
