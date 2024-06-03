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


class MaskedSequenceToSequenceTransformer(nn.Module):
    def __init__(self, src_input_dim, tgt_output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout=0.1, seq_length=60):
        super(MaskedSequenceToSequenceTransformer, self).__init__()

        self.seq_length = seq_length
        self.d_model = d_model

        # Linear layers to project the input features to the model dimension
        self.src_input_linear = nn.Linear(src_input_dim, d_model)
        self.tgt_input_linear = nn.Linear(tgt_output_dim, d_model)
        self.output_linear = nn.Linear(d_model, tgt_output_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_length)

        # Transformer Encoder
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

    def forward(self, src, tgt, feature_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # Apply feature mask to the source input
        if feature_mask is not None:
            src = src * feature_mask

        # Encode the source sequence
        src = self.src_input_linear(src)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Prepare the target sequence
        tgt = self.tgt_input_linear(tgt)
        tgt = self.positional_encoding(tgt)

        # Decode the target sequence
        output = self.transformer_decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_linear(output)

        return output

    def generate(self, src, max_len, src_mask=None):
        # Encode the source sequence
        src = self.src_input_linear(src)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        # Start with the start-of-sequence token
        batch_size = src.size(0)
        tgt = torch.zeros(batch_size, 1, self.d_model).cuda()  # Initialize with zeros or start token in d_model space

        # Initialize the feature mask
        feature_mask = torch.ones(batch_size, 1, self.d_model).cuda()
        feature_mask[:, :, -13:] = 0  # Mask last 13 features initially

        outputs = []

        for i in range(13):
            tgt_step = self.tgt_input_linear(tgt)
            tgt_step = self.positional_encoding(tgt_step)
            tgt_mask = self.generate_square_subsequent_mask(tgt_step.size(1)).cuda()
            output_step = self.transformer_decoder(tgt_step, memory, tgt_mask=tgt_mask)
            next_token = self.output_linear(output_step[:, -1, :])
            outputs.append(next_token.unsqueeze(1))
            next_token_proj = self.tgt_input_linear(next_token).unsqueeze(1)
            tgt = torch.cat((tgt, next_token_proj), dim=1)
            # Update the feature mask to exclude the newly predicted feature
            feature_mask = torch.cat((feature_mask, torch.ones(batch_size, 1, self.d_model).cuda()), dim=1)
            feature_mask[:, :, -(13 - i - 1):] = 0

        return torch.cat(outputs, dim=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.bool()


# Example usage
src_input_dim = 25
tgt_output_dim = 13
d_model = 32
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 64
dropout = 0.1
seq_length = 60

model = MaskedSequenceToSequenceTransformer(src_input_dim, tgt_output_dim, d_model, nhead, num_encoder_layers,
                                            num_decoder_layers, dim_feedforward, dropout, seq_length).cuda()

# Dummy input
src = torch.rand(16, seq_length, src_input_dim).cuda()  # batch_size=16, seq_length=60, src_input_dim=25
tgt = torch.rand(16, seq_length, tgt_output_dim).cuda()  # batch_size=16, seq_length=60, tgt_output_dim=13

output = model(src, tgt)

print(output.shape)  # Should print torch.Size([16, 60, 13])

max_len = 13  # Number of predictions to make
generated_output = model.generate(src, max_len)

print(generated_output.shape)  # Should print torch.Size([16, 13, 13])
