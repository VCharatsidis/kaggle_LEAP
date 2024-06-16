
import numpy as np
import torch


from torch.utils.data import DataLoader
from seq2seq_utils import seq2scalar_32, count_parameters, eval_model, collate_fn
from neural_net.utils import r2_score
import polars as pl
from constants import BATCH_SIZE, LEARNING_RATE, seq_variables_x, \
    scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length, TARGET_WEIGHTS
from transformer_constants import input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, \
    dim_feedforward, dropout

from transformers import get_linear_schedule_with_warmup

train_file = '../data/train_set.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

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

model_name = f'seq2scalar_weighted_32_positional_{min_std}.model'
model = torch.load(f"models/{model_name}")

print(f'The model has {count_parameters(model):,} trainable parameters')
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Example of processing each chunk

val_data = pl.read_csv("../data/validation_set.csv")
val_dataset, _ = seq2scalar_32(True, val_data, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x,
                               scalar_variables_x, seq_variables_y, scalar_variables_y)

print("val dataset:", val_data.shape)
val_loader = DataLoader(val_dataset,
                        batch_size=1000,
                        shuffle=False,
                        )


model.eval()
with torch.no_grad():
    val_loss = torch.zeros([len(TARGET_COLS)], dtype=torch.float64).cuda()
    sum_val_preds = torch.zeros([len(TARGET_COLS)], dtype=torch.float32).cuda()

    for src, tgt in val_loader:
        val_preds = model(src)
        val_preds = val_preds.cpu()

        tgt = tgt.cpu().numpy()
        tgt = ((tgt * std_y) + mean_y)

        val_preds[:, std_y < (1.1 * min_std)] = 0
        val_preds = val_preds.numpy()

        val_preds = ((val_preds * std_y) + mean_y)

        tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
        val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

        ss_res = (tgt - val_preds) ** 2
        score = torch.sum(ss_res, dim=0)

        val_loss += score

    print(val_loss.shape)
    mean_val = val_loss / val_data.shape[0]
    print(mean_val.shape)

    for i in range(len(TARGET_COLS)):
        print(TARGET_COLS[i], round(mean_val[i].item(), 5), round(TARGET_WEIGHTS[i], 5))

    print(mean_val.mean())


