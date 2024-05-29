import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from modified_seq_to_scalars import ModifiedSequenceToScalarTransformer
from seq_to_scalars_transformer import SequenceToScalarTransformer
from seq2scalar.nn_architecture.simple_transformer import SimpleTransformerModel
from seq2scalar.nn_architecture.transformer import TransformerSeq2Seq
from seq2seq_utils import seq2scalar_32, count_parameters, eval_model, collate_fn
from neural_net.utils import r2_score
import polars as pl
from constants import BATCH_SIZE, LEARNING_RATE, seq_variables_x, \
    scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length, ERR
from transformer_constants import input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, \
    dim_feedforward, dropout

from transformers import get_linear_schedule_with_warmup

train_file = '../data/train_set.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

#calc_x_mean_and_std(file='../data/train.csv', FEAT_COLS=FEAT_COLS)

mean_y = np.load('../data/mean_weighted_y.npy')
std_y = np.load('../data/std_weighted_y.npy')

std_y = np.clip(std_y, a_min=1e-12, a_max=None)

print(mean_y.shape, std_y.shape)

if np.any(std_y == 0):
    raise ValueError("std_y contains zero values, which would cause division by zero.")
if np.any(np.isnan(std_y)) or np.any(np.isinf(std_y)):
    raise ValueError("std_y contains NaN or inf values.")

if np.any(np.isnan(mean_y)) or np.any(np.isinf(mean_y)):
    raise ValueError("mean_y contains NaN or inf values.")


mean_x = np.load('../data/mean_x.npy')
std_x = np.load('../data/std_x.npy')

# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

chunk_size = 500000  # Define the size of each batch

model_name = 'seq2scalar_weighted_32.model'

model = torch.load(f"models/{model_name}")
# model = model.double()
# model = ModifiedSequenceToScalarTransformer(input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_length).cuda()

print(f'The model has {count_parameters(model):,} trainable parameters')


print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=6, verbose=False)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=5)

num_training_steps = 20000  # Total number of training steps
num_warmup_steps = 100     # Number of steps to warm up the learning rate

# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=num_warmup_steps,
#                                             num_training_steps=num_training_steps)

min_loss = 10000000000000
# Example of processing each chunk

val_data = pl.read_csv("../data/validation_set.csv")
val_dataset, _ = seq2scalar_32(True, val_data, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x,
                               scalar_variables_x, seq_variables_y, scalar_variables_y)

print("val dataset:", val_data.shape)
val_loader = DataLoader(val_dataset,
                        batch_size=8 * BATCH_SIZE,
                        shuffle=False,
                        # num_workers=4,
                        # pin_memory=True
                        )

patience = 0
epoch = 0
num_epochs = 5

reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
batches = reader.next_batches(20)

# train_sample = pl.read_csv("../data/train_set.csv", n_rows=100000)
# batches = [train_sample]

print("batches:", len(batches), "shapes:", [batch.shape for batch in batches])
start_from = 8
criterion = nn.MSELoss()  # Using MSE for regression
while patience < num_epochs:
    counter = 0
    iterations = 0
    for idx, df in enumerate(batches):
        # if idx < start_from:
        #     continue
        # else:
        #     counter = start_from
        #     start_from = 0

        prep_chunk_time_start = time.time()

        train_dataset, _ = seq2scalar_32(True, df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x,
                                         scalar_variables_x, seq_variables_y, scalar_variables_y)

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  # num_workers=4,
                                  # pin_memory=True
                                  )

        prep_chunk_time_end = time.time()

        print("epoch:", epoch, "chunk:", counter, "prep chunk time:", prep_chunk_time_end - prep_chunk_time_start)

        model.train()
        total_loss = 0
        steps = 0
        train_time_start = time.time()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            # if batch_idx > 500:
            #     break

            optimizer.zero_grad()
            preds = model(src)
            preds[:, std_y < (1.1 * ERR)] = 0

            loss = criterion(preds, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #scheduler.step()

            total_loss += loss.item()

            iterations += 1
            steps += 1

            if (batch_idx + 1) % 100 == 0:
                train_time_end = time.time()
                print("epoch:", epoch, f', chunk: {counter}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}', "iterations:", iterations, "time:", train_time_end - train_time_start)
                total_loss = 0  # Reset the loss for the next steps
                steps = 0  # Reset step count

        patience, min_loss = eval_model(model, val_loader, min_loss,
                                        patience, epoch, counter, iterations, model_name, std_y)

        for param_group in optimizer.param_groups:
            print(f"End of Epoch {epoch}, Learning Rate: {param_group['lr']}")

        print()
        counter += 1

    epoch += 1
