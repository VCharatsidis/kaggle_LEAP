import numpy as np
import torch
import math
import torch.nn as nn
import random

from constants import LEARNING_RATE
from seq2seq import SequenceToSequenceTransformer
from seq2seq_utils import mean_and_flatten

# Example usage
src_input_dim = 25
tgt_input_dim = 14
output_dim = 14
d_model = 512
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


output = model(src, tgt)
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer.zero_grad()

output = mean_and_flatten(output)
tgt = mean_and_flatten(tgt)

mean_y = np.load('../../data/mean_weighted_y.npy')
std_y = np.load('../../data/std_weighted_y.npy')

min_std = 1e-12
std_y = np.clip(std_y, a_min=min_std, a_max=None)
output[:, std_y < (1.1 * min_std)] *= 0

criterion = nn.MSELoss()  # Using MSE for regression
loss = criterion(output, tgt)

loss.backward()

print(output.shape)  # Should print torch.Size([16, 60, 14])

max_len = 60
generated_output = model.generate(src, max_len)

print(generated_output.shape)  # Should print torch.Size([16, 60, 14])
