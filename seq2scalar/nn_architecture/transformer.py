import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Perform linear operation and split into num_heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)

        # Concatenate heads and put through final linear layer
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(x.shape, self.pe.shape)
        return x + self.pe.permute(1, 0, 2).expand(x.shape[0], -1, -1)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention and add & norm
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.layer_norm1(src + self.dropout(src2))

        # Feed forward and add & norm
        src2 = self.feed_forward(src)
        src = self.layer_norm2(src + self.dropout(src2))

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention and add & norm
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.layer_norm1(tgt + self.dropout(tgt2))

        # Cross-attention and add & norm
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.layer_norm2(tgt + self.dropout(tgt2))

        # Feed forward and add & norm
        tgt2 = self.feed_forward(tgt)
        tgt = self.layer_norm3(tgt + self.dropout(tgt2))

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim_src, output_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(TransformerSeq2Seq, self).__init__()

        self.encoder = TransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout)

        self.input_fc_src = nn.Linear(input_dim_src, d_model)
        self.input_fc_tgt = nn.Linear(output_dim, d_model)
        self.output_fc = nn.Linear(d_model, output_dim)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        self.d_model = d_model

    def forward(self, src, tgt):
        src = self.input_fc_src(src) * math.sqrt(self.d_model)
        tgt = self.input_fc_tgt(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.output_fc(output)
        return output

    def generate(self, src, max_len):
        src = self.input_fc_src(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.encoder(src)

        # Prepare initial input (e.g., zeros or start token)
        tgt = torch.zeros(src.size(0), 1, self.d_model).to(src.device)  # Start with a single time step
        outputs = []

        for _ in range(max_len):
            tgt_pos_enc = self.pos_decoder(tgt)
            output = self.decoder(tgt_pos_enc, memory)
            output = self.output_fc(output[:, -1, :])  # Get the last time step and reshape to [batch_size, output_dim]

            outputs.append(output.unsqueeze(1))  # Add a new dimension to match [batch_size, 1, output_dim]
            new_tgt = self.input_fc_tgt(output).unsqueeze(1)  # Transform output back to tgt space and add dimension
            tgt = torch.cat([tgt, new_tgt], dim=1)  # Append the generated output to the target sequence

        outputs = torch.cat(outputs, dim=1)  # Concatenate along the sequence dimension
        return outputs


