import numpy as np
import torch
import math
import torch.nn as nn
import random

from constants import LEARNING_RATE
from seq2seq_utils import mean_and_flatten


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
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

        self.positional_encoding = PositionalEncoding(d_model, seq_length=src_seq_length)

        self.src_seq_length = src_seq_length
        #self.tgt_seq_length = tgt_seq_length
        self.d_model = d_model
        self.output_dim = tgt_input_dim

        # Linear layers to project the input features to the model dimension
        self.src_input_linear = nn.Linear(src_input_dim, d_model)
        #self.tgt_input_linear = nn.Linear(tgt_input_dim, d_model)
        self.output_linear = nn.Linear(d_model, tgt_input_dim)
        self.output_to_dmodel = nn.Linear(tgt_input_dim, d_model)

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

    def forward(self, src, tgt, teacher_forcing_ratio=0.35):
        # Encode the source sequence
        memory = self.transformer_encoder(self.positional_encoding(self.src_input_linear(src)))

        # Initialize the output tensor
        #tgt = self.tgt_input_linear(tgt)

        batch_size, seq_len, _ = src.size()
        outputs = torch.zeros(batch_size, seq_len, self.output_dim).cuda()
        #teacher_forcing_decisions = np.random.rand(seq_len) < teacher_forcing_ratio

        # Pre-allocate memory for decoder inputs and set the initial input
        decoder_input = torch.zeros(batch_size, seq_len, self.d_model).cuda()

        for t in range(0, seq_len - 1):
            pos_decoder_input = self.positional_encoding(decoder_input[:, :(t+1), :])
            output_step = self.transformer_decoder(pos_decoder_input, memory)
            next_token = self.output_linear(output_step[:, -1, :])
            outputs[:, t, :] = next_token

            # if teacher_forcing_decisions[t]:
            #     next_input = tgt[:, t, :]  # Use the ground truth token
            # else:
            #     next_input = self.output_to_dmodel(next_token)  # Use the predicted token
            next_input = self.output_to_dmodel(next_token)
            decoder_input[:, (t+1), :] = next_input

        pos_decoder_input = self.positional_encoding(decoder_input[:, :seq_len, :])
        output_step = self.transformer_decoder(pos_decoder_input, memory)
        outputs[:, seq_len - 1, :] = self.output_linear(output_step[:, -1, :])

        return outputs

    def generate(self, src, seq_len=60):
        # Encode the source sequence
        memory = self.transformer_encoder(self.positional_encoding(self.src_input_linear(src)))

        # Start with the start-of-sequence token
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, seq_len, self.output_dim).cuda()
        decoder_inputs = torch.zeros(batch_size, seq_len, self.d_model).cuda()

        for t in range(0, seq_len - 1):
            pos_dec_input = self.positional_encoding(decoder_inputs[:, :(t+1), :])
            output_step = self.transformer_decoder(pos_dec_input, memory)
            next_token = self.output_linear(output_step[:, -1, :])
            outputs[:, t, :] = next_token
            decoder_inputs[:, (t+1), :] = self.output_to_dmodel(next_token)

        decoder_input_last = self.positional_encoding(decoder_inputs[:, :seq_len, :])
        output_step = self.transformer_decoder(decoder_input_last, memory)
        outputs[:, seq_len - 1, :] = self.output_linear(output_step[:, -1, :])

        return outputs


class SequenceToSequence_no_teacher(nn.Module):
    def __init__(self, src_input_dim, tgt_input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, src_seq_length=60, tgt_seq_length=60):
        super(SequenceToSequence_no_teacher, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, seq_length=src_seq_length)

        self.src_seq_length = src_seq_length
        #self.tgt_seq_length = tgt_seq_length
        self.d_model = d_model
        self.output_dim = tgt_input_dim

        # Linear layers to project the input features to the model dimension
        self.src_input_linear = nn.Linear(src_input_dim, d_model)
        #self.tgt_input_linear = nn.Linear(tgt_input_dim, d_model)
        self.output_linear = nn.Linear(d_model, tgt_input_dim)
        self.output_to_dmodel = nn.Linear(tgt_input_dim, d_model)

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

    def forward(self, src):
        # Encode the source sequence
        memory = self.transformer_encoder(self.positional_encoding(self.src_input_linear(src)))

        batch_size, seq_len, _ = src.size()
        outputs = torch.zeros(batch_size, seq_len, self.output_dim).cuda()

        # Pre-allocate memory for decoder inputs and set the initial input
        decoder_input = torch.zeros(batch_size, seq_len, self.d_model).cuda()

        for t in range(0, seq_len - 1):
            pos_decoder_input = self.positional_encoding(decoder_input[:, :(t+1), :])
            output_step = self.transformer_decoder(pos_decoder_input, memory)
            next_token = self.output_linear(output_step[:, -1, :])
            outputs[:, t, :] = next_token
            next_input = self.output_to_dmodel(next_token)
            decoder_input[:, (t+1), :] = next_input

        pos_decoder_input = self.positional_encoding(decoder_input[:, :seq_len, :])
        output_step = self.transformer_decoder(pos_decoder_input, memory)
        outputs[:, seq_len - 1, :] = self.output_linear(output_step[:, -1, :])

        return outputs

    def generate(self, src, seq_len=60):
        # Encode the source sequence
        memory = self.transformer_encoder(self.positional_encoding(self.src_input_linear(src)))

        # Start with the start-of-sequence token
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, seq_len, self.output_dim).cuda()
        decoder_inputs = torch.zeros(batch_size, seq_len, self.d_model).cuda()

        for t in range(0, seq_len - 1):
            pos_dec_input = self.positional_encoding(decoder_inputs[:, :(t+1), :])
            output_step = self.transformer_decoder(pos_dec_input, memory)
            next_token = self.output_linear(output_step[:, -1, :])
            outputs[:, t, :] = next_token
            decoder_inputs[:, (t+1), :] = self.output_to_dmodel(next_token)

        decoder_input_last = self.positional_encoding(decoder_inputs[:, :seq_len, :])
        output_step = self.transformer_decoder(decoder_input_last, memory)
        outputs[:, seq_len - 1, :] = self.output_linear(output_step[:, -1, :])

        return outputs