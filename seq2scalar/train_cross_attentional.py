import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from torch.utils.data import DataLoader

from cross_attention import CrossAttentionModel
from neural_net.utils import r2_score
from seq2seq_utils import seq2scalar_32, count_parameters, collate_fn
from constants import seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length, \
    input_variable_order, TARGET_WEIGHTS

train_file = '../data/train_set_2.csv'
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

model_name = f'cross_attentional_{min_std}_2.model'

feature_dim = 25
d_model = 256
nhead = 2
num_encoder_layers = 2
dim_feedforward = 256
output_dim = 368
dropout = 0.1
LEARNING_RATE = 1e-5
BATCH_SIZE = 1024
#model = CrossAttentionModel(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()

model = torch.load(f"models/{model_name}")

print(f'The model has {count_parameters(model):,} trainable parameters')
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

num_training_steps = 20000  # Total number of training steps
num_warmup_steps = 100     # Number of steps to warm up the learning rate


val_data = pl.read_csv("../data/validation_set_2.csv")
val_dataset, _ = seq2scalar_32(True, val_data, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x,
                               scalar_variables_x, seq_variables_y, scalar_variables_y)

print("val dataset:", val_data.shape)
val_loader = DataLoader(val_dataset,
                        batch_size= BATCH_SIZE,
                        shuffle=False,
                        )


# Map the desired order of variables to their indices in the original tensor
variable_order_indices = [
    seq_variables_x.index(var) if var in seq_variables_x else len(seq_variables_x) + scalar_variables_x.index(var)
    for var in input_variable_order]


def get_feature_data(tensor_data, variable_order_indices):
    # Reshape and reorder to N, 25, 60
    tensor_data = tensor_data.permute(0, 2, 1)  # Change to N, 25, 60
    tensor_data_ordered = tensor_data[:, variable_order_indices, :]  # Reorder the second dimension as per the desired order

    # The tensor_data_ordered is now in the shape N, 25, 60 with variables ordered as specified
    return tensor_data_ordered


def get_val_loss_cross_attention(weighted):
    model.eval()
    mask = std_y < (1.1 * min_std)
    with torch.no_grad():
        val_loss = 0
        val_time_start = time.time()
        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds = val_preds.cpu()

            tgt = tgt.cpu().numpy()
            if not weighted:
                tgt = ((tgt * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                tgt = (tgt * std_y) + mean_y

            val_preds[:, mask] = 0
            val_preds = val_preds.numpy()

            if not weighted:
                val_preds = ((val_preds * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                val_preds = (val_preds * std_y) + mean_y

            tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
            val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

            val_loss += r2_score(val_preds, tgt).item()

        val_time_end = time.time()

        avg_val_loss = val_loss / len(val_loader)

        print("time:", val_time_end - val_time_start, "val loss:", avg_val_loss)
        return avg_val_loss

min_loss = get_val_loss_cross_attention(weighted=True)
# Example of processing each chunk

patience = 0
epoch = 0
num_epochs = 5

reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
batches = reader.next_batches(20)



def eval_cross_attention(min_std, weighted, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, mean_y, std_y):
    model.eval()
    mask = std_y < (1.1 * min_std)
    with torch.no_grad():
        val_loss = 0
        val_time_start = time.time()
        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds = val_preds.cpu()

            tgt = tgt.cpu().numpy()
            if not weighted:
                tgt = ((tgt * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                tgt = (tgt * std_y) + mean_y

            val_preds[:, mask] = 0
            val_preds = val_preds.numpy()

            if not weighted:
                val_preds = ((val_preds * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                val_preds = (val_preds * std_y) + mean_y

            tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
            val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

            val_loss += r2_score(val_preds, tgt).item()

        val_time_end = time.time()

        avg_val_loss = val_loss / len(val_loader)
        #scheduler.step()  # Adjust learning rate

        if avg_val_loss < min_loss:
            print("epoch:", epoch, "model saved", "chunk:", counter, "iterations:", iterations, "val loss:",
                  avg_val_loss, "time:", val_time_end - val_time_start)
            torch.save(model, f"models/{model_name}")
            min_loss = avg_val_loss
            patience = 0
        else:
            print("epoch:", epoch, f"No improvement in validation loss for {patience} epochs.", "chunk:", counter,
                  "iterations:", iterations, "val loss:", avg_val_loss, "time:", val_time_end - val_time_start)
            patience += 1

    return patience, min_loss


print("batches:", len(batches), "shapes:", [batch.shape for batch in batches])

criterion = nn.MSELoss()  # Using MSE for regression


while patience < num_epochs:
    if epoch == 0:
        counter = 1
    else:
        counter = 0

    iterations = 0
    for idx, df in enumerate(batches):
        if idx < counter:
            continue

        prep_chunk_time_start = time.time()

        train_dataset, _ = seq2scalar_32(True, df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x,
                                         scalar_variables_x, seq_variables_y, scalar_variables_y)

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  )

        prep_chunk_time_end = time.time()

        print("epoch:", epoch, "chunk:", counter, "prep chunk time:", prep_chunk_time_end - prep_chunk_time_start)

        total_loss = 0
        steps = 0
        train_time_start = time.time()
        mask = std_y < (1.1 * min_std)

        model.train()

        for batch_idx, (src, tgt) in enumerate(train_loader):

            optimizer.zero_grad()
            src_2 = get_feature_data(src, variable_order_indices)
            preds = model(src, src_2)
            preds[:, mask] *= 0

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

        patience, min_loss = eval_cross_attention(min_std, True, model, val_loader, min_loss,
                                        patience, epoch, counter, iterations, model_name, mean_y, std_y)

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, end of chunk {idx}, Learning Rate: {param_group['lr']}")

        print()
        counter += 1

    epoch += 1
