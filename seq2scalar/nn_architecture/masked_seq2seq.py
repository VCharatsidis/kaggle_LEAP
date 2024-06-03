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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerWithoutTargetInput(nn.Module):
    def __init__(self, src_input_dim, tgt_output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout=0.1, seq_length=60):
        super(TransformerWithoutTargetInput, self).__init__()

        self.seq_length = seq_length
        self.d_model = d_model

        # Linear layers to project the input features to the model dimension
        self.src_input_linear = nn.Linear(src_input_dim, d_model)
        self.output_linear = nn.Linear(d_model, tgt_output_dim)
        self.output_to_dmodel = nn.Linear(tgt_output_dim, d_model)

        # Positional encoding
        self.src_positional_encoding = PositionalEncoding(d_model, max_len=seq_length)
        self.tgt_positional_encoding = PositionalEncoding(d_model, max_len=seq_length)

        # Transformer Encoder to process the sequence
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True),
            num_layers=num_encoder_layers
        )

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True),
            num_layers=num_decoder_layers
        )

    def forward(self, src, src_key_padding_mask=None):
        # Encode the source sequence
        src = self.src_input_linear(src)
        src = self.src_positional_encoding(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Initialize the output tensor
        tgt = torch.zeros(src.size(0), 1, self.d_model).cuda()  # Start token in d_model space
        outputs = []

        for _ in range(self.seq_length):
            tgt_step = self.tgt_positional_encoding(tgt)
            tgt_mask = self.generate_square_subsequent_mask(tgt_step.size(1)).cuda()
            output_step = self.transformer_decoder(tgt_step, memory, tgt_mask=tgt_mask,
                                                   memory_key_padding_mask=src_key_padding_mask)
            next_token = self.output_linear(output_step[:, -1, :])
            outputs.append(next_token.unsqueeze(1))
            next_token_proj = self.output_to_dmodel(next_token).unsqueeze(1)
            tgt = torch.cat((tgt, next_token_proj), dim=1)

        return torch.cat(outputs, dim=1)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.bool()


# # Example usage
# src_input_dim = 25
# tgt_output_dim = 13
# d_model = 32
# nhead = 4
# num_encoder_layers = 2
# num_decoder_layers = 2
# dim_feedforward = 64
# dropout = 0.1
# seq_length = 60
#
# model = TransformerWithoutTargetInput(src_input_dim, tgt_output_dim, d_model, nhead, num_encoder_layers,
#                                       num_decoder_layers, dim_feedforward, dropout, seq_length).cuda()
#
# # Dummy input
# src = torch.rand(16, seq_length, src_input_dim).cuda()  # batch_size=16, seq_length=60, src_input_dim=25
# output = model(src)
#
# print(output.shape)  # Should print torch.Size([16, 60, 13])
