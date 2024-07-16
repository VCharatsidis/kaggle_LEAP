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


class Conv1DBlock(nn.Module):
    def __init__(self, dim, ksize, drop_rate=0):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(dim, dim, ksize, padding=ksize // 2)
        self.bn = nn.BatchNorm1d(dim)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Hoyso_Transformer(nn.Module):
    def __init__(self, seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0):
        super(Hoyso_Transformer, self).__init__()

        self.conv_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.num_layers = num_encoder_layers

        ksize = 5
        for _ in range(num_encoder_layers):
            self.conv_blocks.append(Conv1DBlock(seq_length, ksize, dropout))
            # self.conv_blocks.append(Conv1DBlock(seq_length, ksize, dropout))
            # self.conv_blocks.append(Conv1DBlock(seq_length, ksize, dropout))

            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                       dropout=dropout, batch_first=True, activation=nn.GELU())
            self.transformer_blocks.append(nn.TransformerEncoder(encoder_layer, num_layers=1))

        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True, activation=nn.GELU()),
            num_layers=num_encoder_layers
        )

        # Linear layers to adapt dimensions
        self.input_adapter1 = nn.Linear(feature_dim, d_model)
        self.input_adapter2 = nn.Linear(seq_length, d_model)

        self.learned_encoding_1 = LearnedPositionalEncoding(d_model, max_len=feature_dim)

        self.cross_attention_1 = CrossAttention(d_model, nhead)
        self.cross_attention_2 = CrossAttention(d_model, nhead)

        # self.gru = nn.GRU(256, 1024, batch_first=True, bidirectional=True)

        # # Output layer
        self.output_linear = nn.Linear(85 * d_model, output_dim)  # Adjust output layer size

    def forward(self, src1):
        # Reshape and reorder to N, 25, 60
        src2 = src1.permute(0, 2, 1)  # Change to N, 25, 60
        # Prepare inputs

        src1 = self.input_adapter1(src1)  # N, 60, 25 -> N, 60, d_model
        src2 = self.input_adapter2(src2)  # N, 25, 60 -> N, 25, d_model

        for i in range(self.num_layers):
            src1 = self.conv_blocks[i](src1)
            # src1 = self.conv_blocks[i * 3 + 1](src1)
            # src1 = self.conv_blocks[i * 3 + 2](src1)

            src1 = self.transformer_blocks[i](src1)

        # # Add positional encoding
        src2 = self.learned_encoding_1(src2)
        encoded2 = self.encoder2(src2)

        # Apply cross-attention
        cross_attended1 = self.cross_attention_1(src1, encoded2, encoded2).flatten(start_dim=1)
        cross_attended2 = self.cross_attention_2(encoded2, src1, src1).flatten(start_dim=1)

        # Combine outputs from both paths
        combined = torch.cat([cross_attended1, cross_attended2], dim=1)

        output = self.output_linear(combined)
        return output


# Testing the model
def model_test():
    N = 256  # Batch size
    src1 = torch.randn(N, 60, 25)  # N, Layers, Features

    model = Hoyso_Transformer(
        seq_length=60,
        feature_dim=25,
        d_model=256,
        nhead=8,
        num_encoder_layers=7,
        dim_feedforward=512,
        output_dim=368,
        dropout=0,
    )

    output = model(src1)
    print("Output Shape:", output.shape)


