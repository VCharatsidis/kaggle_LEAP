import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from seq2seq import SequenceToSequenceTransformer
from seq2seq_utils import count_parameters, eval_model, collate_fn, seq2seq_32, eval_model_seq_2_seq, mean_and_flatten, \
    eval_generate_model_seq_2_seq
import polars as pl
from constants import BATCH_SIZE, LEARNING_RATE, seq_variables_x, \
    scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length
from transformer_constants import input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, \
    dim_feedforward, dropout, scalar_vars_num, vector_vars_num

from transformers import get_linear_schedule_with_warmup

train_file = '../data/train_set.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

#calc_x_mean_and_std(file='../data/train.csv', FEAT_COLS=FEAT_COLS)

mean_y = np.load('../data/mean_weighted_y.npy')
std_y = np.load('../data/std_weighted_y.npy')

min_std = 1e-12
std_y = np.clip(std_y, a_min=min_std, a_max=None)

print(mean_y.shape, std_y.shape)

if np.any(std_y == 0):
    raise ValueError("std_y contains zero values, which would cause division by zero.")
if np.any(np.isnan(std_y)) or np.any(np.isinf(std_y)):
    raise ValueError("std_y contains NaN or inf values.")

if np.any(np.isnan(mean_y)) or np.any(np.isinf(mean_y)):
    raise ValueError("mean_y contains NaN or inf values.")


mean_x = np.load('../data/mean_x.npy')
std_x = np.load('../data/std_x.npy')

std_x = np.clip(std_x, a_min=min_std, a_max=None)

# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

chunk_size = 500000  # Define the size of each batch

model_name = f'seq2seq_weighted_32_positional_{min_std}.model'

#model = torch.load(f"models/{model_name}")

src_input_dim = 25
tgt_input_dim = 14
model = SequenceToSequenceTransformer(src_input_dim, tgt_input_dim, d_model, nhead, num_encoder_layers,
                                      dim_feedforward, dropout=0.1, src_seq_length=60, tgt_seq_length=60).cuda()
#
# torch.save(model, f"models/{model_name}")

print(f'The model has {count_parameters(model):,} trainable parameters')
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

num_training_steps = 20000  # Total number of training steps
num_warmup_steps = 100     # Number of steps to warm up the learning rate

# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=num_warmup_steps,
#                                             num_training_steps=num_training_steps)

min_loss = 10000000000000
# Example of processing each chunk

val_data = pl.read_csv("../data/validation_set.csv")
val_dataset, _ = seq2seq_32(min_std, val_data, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x,
                               scalar_variables_x, seq_variables_y, scalar_variables_y)

print("val dataset:", val_data.shape)
val_loader = DataLoader(val_dataset,
                        batch_size=2*BATCH_SIZE,
                        shuffle=True,
                        # num_workers=4,
                        # pin_memory=True
                        )


patience, min_loss = eval_generate_model_seq_2_seq(5, min_std, True, model, val_loader, min_loss, 0, 0, 0, 0, model_name, mean_y, std_y)

patience = 0
epoch = 0
num_epochs = 5

reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
batches = reader.next_batches(20)

# train_sample = pl.read_csv("../data/train_set.csv", n_rows=100000)
# batches = [train_sample]

#patience, min_loss = eval_generate_model_seq_2_seq(min_std, True, model, val_loader, min_loss, patience, epoch, 0, 0, model_name, mean_y, std_y)

up_to = 400
print("batches:", len(batches), "shapes:", [batch.shape for batch in batches])
start_from = 8
criterion = nn.MSELoss()  # Using MSE for regression
while patience < num_epochs:
    counter = 0
    iterations = 0
    for idx, df in enumerate(batches):

        prep_chunk_time_start = time.time()

        train_dataset, _ = seq2seq_32(min_std, df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x,
                                         scalar_variables_x, seq_variables_y, scalar_variables_y)

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  )

        prep_chunk_time_end = time.time()

        print("epoch:", epoch, "chunk:", counter, "prep chunk time:", prep_chunk_time_end - prep_chunk_time_start)

        model.train()
        total_loss = 0
        steps = 0
        train_time_start = time.time()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            if (batch_idx % 10 == 0) and (steps > 0):
                print("batch_idx:", batch_idx, "loss:", total_loss / steps)

            optimizer.zero_grad()
            preds = model(src, tgt)

            preds = mean_and_flatten(preds)
            tgt = mean_and_flatten(tgt)
            preds[:, std_y < (1.1 * min_std)] *= 0

            loss = criterion(preds, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            iterations += 1
            steps += 1

            if (batch_idx + 1) % 100 == 0:
                train_time_end = time.time()
                print("epoch:", epoch, f', chunk: {counter}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}', "iterations:", iterations, "time:", train_time_end - train_time_start)
                total_loss = 0  # Reset the loss for the next steps
                steps = 0  # Reset step count

        patience, min_loss = eval_generate_model_seq_2_seq(up_to, min_std, True, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, mean_y, std_y)

        # patience, min_loss = eval_model_seq_2_seq(min_std, True, model, val_loader, min_loss,
        #                                 patience, epoch, counter, iterations, model_name, mean_y, std_y)

        for param_group in optimizer.param_groups:
            print(f"End of Epoch {epoch}, Learning Rate: {param_group['lr']}")

        print()
        counter += 1

    epoch += 1
