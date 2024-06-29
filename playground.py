import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0

        self.d_k = d_model // nhead
        self.nhead = nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()

        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_length, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_length, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_length, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)

        # Final linear layer
        output = self.out(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="gelu", norm_first=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x):
        x = self.self_attn(x)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.feed_forward(x)
        return self.dropout2(x)


# Define parameters
d_model = 512
nhead = 8
dim_feedforward = 2048
dropout = 0.1

# Create a manually implemented Transformer Encoder Layer
manual_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                               dropout=dropout, activation='gelu', norm_first=False)

# Create a PyTorch Transformer Encoder Layer
pytorch_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='gelu', batch_first=True,
                                                   norm_first=False)

# Initialize both layers with the same weights for comparison
with torch.no_grad():
    manual_encoder_layer.self_attn.q_linear.weight.copy_(pytorch_encoder_layer.self_attn.in_proj_weight[:d_model, :])
    manual_encoder_layer.self_attn.k_linear.weight.copy_(
        pytorch_encoder_layer.self_attn.in_proj_weight[d_model:2 * d_model, :])
    manual_encoder_layer.self_attn.v_linear.weight.copy_(
        pytorch_encoder_layer.self_attn.in_proj_weight[2 * d_model:, :])
    manual_encoder_layer.self_attn.q_linear.bias.copy_(pytorch_encoder_layer.self_attn.in_proj_bias[:d_model])
    manual_encoder_layer.self_attn.k_linear.bias.copy_(
        pytorch_encoder_layer.self_attn.in_proj_bias[d_model:2 * d_model])
    manual_encoder_layer.self_attn.v_linear.bias.copy_(pytorch_encoder_layer.self_attn.in_proj_bias[2 * d_model:])
    manual_encoder_layer.self_attn.out.weight.copy_(pytorch_encoder_layer.self_attn.out_proj.weight)
    manual_encoder_layer.self_attn.out.bias.copy_(pytorch_encoder_layer.self_attn.out_proj.bias)
    manual_encoder_layer.feed_forward.linear1.weight.copy_(pytorch_encoder_layer.linear1.weight)
    manual_encoder_layer.feed_forward.linear1.bias.copy_(pytorch_encoder_layer.linear1.bias)
    manual_encoder_layer.feed_forward.linear2.weight.copy_(pytorch_encoder_layer.linear2.weight)
    manual_encoder_layer.feed_forward.linear2.bias.copy_(pytorch_encoder_layer.linear2.bias)
    manual_encoder_layer.norm1.weight.copy_(pytorch_encoder_layer.norm1.weight)
    manual_encoder_layer.norm1.bias.copy_(pytorch_encoder_layer.norm1.bias)
    manual_encoder_layer.norm2.weight.copy_(pytorch_encoder_layer.norm2.weight)
    manual_encoder_layer.norm2.bias.copy_(pytorch_encoder_layer.norm2.bias)

# Example input
batch_size = 32
sequence_length = 10
example_input = torch.rand(batch_size, sequence_length, d_model)

# Set both models to evaluation mode to avoid dropout randomness
manual_encoder_layer.eval()
pytorch_encoder_layer.eval()

# Pass the input through both encoder layers
manual_output = manual_encoder_layer(example_input)
pytorch_output = pytorch_encoder_layer(example_input)

# Compare the outputs
print("Manual output shape:", manual_output.shape)
print("PyTorch output shape:", pytorch_output.shape)
print("Difference:", torch.abs(manual_output - pytorch_output).max().item())
