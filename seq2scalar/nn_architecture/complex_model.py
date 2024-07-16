import torch
import torch.nn as nn
import math


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=25):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, query, key, value):
        # Cross-attention: query attends to key and value
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class Complex_Transformer(nn.Module):
    def __init__(self, seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0):
        super(Complex_Transformer, self).__init__()

        # Encoders
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True, activation=nn.GELU()),
            num_layers=num_encoder_layers
        )
        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True, activation=nn.GELU()),
            num_layers=num_encoder_layers
        )

        # Linear layers to adapt dimensions
        self.input_adapter1 = nn.Linear(feature_dim, d_model)
        self.input_adapter2 = nn.Linear(seq_length, d_model)

        # Positional encoding is the same for both
        self.positional_encoding_1 = PositionalEncoding(d_model, max_len=seq_length)
        self.learned_encoding_1 = LearnedPositionalEncoding(d_model, max_len=feature_dim)

        self.conv1d_level_wise = nn.Conv1d(seq_length, 60, kernel_size=5, padding='same')
        self.conv1d_dim_wise = nn.Conv1d(feature_dim, 60, kernel_size=5, padding='same')

        self.cross_attention_1 = CrossAttention(d_model, nhead)
        self.cross_attention_2 = CrossAttention(d_model, nhead)

        # self.gru = nn.GRU(256, 1024, batch_first=True, bidirectional=True)

        # # Output layer
        self.output_linear = nn.Linear(30720, output_dim)  # Adjust output layer size

    def forward(self, src1):
        # Reshape and reorder to N, 25, 60
        src2 = src1.permute(0, 2, 1)  # Change to N, 25, 60
        # Prepare inputs

        src1 = self.input_adapter1(src1)  # N, 60, 25 -> N, 60, d_model
        src2 = self.input_adapter2(src2)  # N, 25, 60 -> N, 25, d_model

        # # Add positional encoding
        # src1 = self.positional_encoding_1(src1)
        # src2 = self.learned_encoding_1(src2)

        src1 = self.conv1d_level_wise(src1)
        src2 = self.conv1d_dim_wise(src2)

        # Encode with transformers
        encoded1 = self.encoder1(src1)
        encoded2 = self.encoder2(src2)

        # Apply cross-attention
        cross_attended1 = self.cross_attention_1(encoded1, encoded2, encoded2).flatten(start_dim=1)
        cross_attended2 = self.cross_attention_2(encoded2, encoded1, encoded1).flatten(start_dim=1)

        # Combine outputs from both paths
        combined = torch.cat([cross_attended1, cross_attended2], dim=1)
        #
        # _, h_n = self.gru(combined)  # [1, batch_size, gru_hidden_dim]
        # h_n = h_n.squeeze(0)  # [batch_size, gru_hidden_dim]
        #
        # reshaped_tensor = h_n.permute(1, 0, 2).reshape(combined.shape[0], -1)

        # print(h_n.shape)
        output = self.output_linear(combined)
        return output


# Testing the model
def model_test():
    N = 5  # Batch size
    src1 = torch.randn(N, 60, 25)  # N, Layers, Features

    model = Complex_Transformer(
        seq_length=60,
        feature_dim=25,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        output_dim=368,
        dropout=0.1,
    )

    output = model(src1)
    print("Output Shape:", output.shape)


# model_test()
