import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from torch.utils.data import DataLoader

from cross_attention import CrossAttentionModel
from neural_net.utils import r2_score
from seq2seq_utils import seq2scalar_32, count_parameters, collate_fn, seq2scalar_custom_norm, \
    seq2scalar_custom_norm_weightless, get_feature_data, variable_order_indices
from constants import seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length, \
    input_variable_order, TARGET_WEIGHTS

from transformers import get_linear_schedule_with_warmup

train_file = '../data/train_set.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

import pickle

# Define the path to your pickle file
pickle_file_path = 'transforms_weightless.pickle'

# Load the pickle file
with open(pickle_file_path, 'rb') as handle:
    transforms = pickle.load(handle)


log_shifts = []
logs = []
for t in TARGET_COLS:
    if t not in transforms.keys():
        log_shifts.append(0)
        continue

    type = transforms[t][0]
    shift = transforms[t][2]

    index = TARGET_COLS.index(t)
    if type == "log":
        log_shifts.append(shift)
        logs.append(index)
    else:
        log_shifts.append(0)

    #print("Target", index, t, transforms[t])


print("logs:", logs)

min_std = 1e-12


# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

chunk_size = 100000  # Define the size of each batch

start_from = 66
if start_from > 0:
    print("WARNING: Starting from chunk:", start_from)

feature_dim = 25
d_model = 256
nhead = 8
num_encoder_layers = 6
dim_feedforward = 256
output_dim = 368
dropout = 0.1
LEARNING_RATE = 4.8e-5
BATCH_SIZE = 256
#model = CrossAttentionModel(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()

model_name = f'cross_attentional_{min_std}_nhead_{nhead}_enc_l_{num_encoder_layers}_d_{d_model}_custom_norm_weightless_weighted_train.model'

model = torch.load(f"models/{model_name}")


print(f'The model has {count_parameters(model):,} trainable parameters')
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

num_warmup_steps = 500
num_training_steps = 100000

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

min_std = 1e-12
val_data = pl.read_csv("../data/validation_set.csv")
val_dataset, _, stds, means, shifts = seq2scalar_custom_norm_weightless(min_std, val_data, FEAT_COLS, TARGET_COLS, transforms, seq_variables_x, scalar_variables_x)

# Convert the lists into numpy arrays
means = np.array(means)
stds = np.array(stds)

print("val dataset:", val_data.shape)
val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        )


Targets_norm = torch.tensor(TARGET_WEIGHTS, dtype=torch.float64).cuda()

torch_mean = torch.tensor(means, dtype=torch.float64).cuda()
torch_std = torch.tensor(stds, dtype=torch.float64).cuda()
torch_log_shifts = torch.tensor(log_shifts, dtype=torch.float64).cuda()

np_log_shifts = np.array(log_shifts)

def error_per_target_var(model, means, stds):
    model.eval()
    with torch.no_grad():
        val_loss = torch.zeros([len(TARGET_COLS)], dtype=torch.float64).cuda()
        mask = stds < (1.1 * min_std)

        log_mask = np.zeros(368, dtype=bool)
        log_mask[logs] = True

        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds[:, mask] = 0

            val_preds = val_preds.cpu().numpy()
            val_preds = (val_preds * stds + means)

            tgt = tgt.cpu().numpy()
            tgt = (tgt * stds + means)

            exp_preds = np.exp(val_preds[:, log_mask]) - np_log_shifts[log_mask]
            exp_tgt = np.exp(tgt[:, log_mask]) - np_log_shifts[log_mask]

            val_preds[:, log_mask] = exp_preds
            tgt[:, log_mask] = exp_tgt

            tgt = tgt * TARGET_WEIGHTS
            val_preds = val_preds * TARGET_WEIGHTS

            tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
            val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

            ss_res = (tgt - val_preds) ** 2
            score = torch.sum(ss_res, dim=0)

            val_loss += score

        mean_val = val_loss / val_data.shape[0]

        for i in range(len(TARGET_COLS)):
            print(TARGET_COLS[i], round(mean_val[i].item(), 5), round(TARGET_WEIGHTS[i], 5))

        print(mean_val.mean())


def get_val_loss_cross_attention_weightless(means, stds):
    model.eval()
    mask = stds < (1.1 * min_std)
    with torch.no_grad():
        all_preds = np.zeros((1000000, 368))
        all_targets = np.zeros((1000000, 368))
        val_time_start = time.time()
        start_idx = 0

        log_mask = np.zeros(368, dtype=bool)
        log_mask[logs] = True

        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds[:, mask] = 0

            val_preds = val_preds.cpu().numpy()
            val_preds = (val_preds * stds + means)

            tgt = tgt.cpu().numpy()
            tgt = (tgt * stds + means)

            val_preds[:, log_mask] = np.exp(val_preds[:, log_mask]) - np_log_shifts[log_mask]
            tgt[:, log_mask] = np.exp(tgt[:, log_mask]) - np_log_shifts[log_mask]

            batch_size = tgt.shape[0]
            all_targets[start_idx: start_idx + batch_size] = tgt * TARGET_WEIGHTS
            all_preds[start_idx: start_idx + batch_size] = val_preds * TARGET_WEIGHTS
            start_idx += batch_size

        # Calculate R² score
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        avg_val_loss = ss_res / ss_tot

        val_time_end = time.time()

        print("time:", val_time_end - val_time_start, "val loss:", avg_val_loss)
        return avg_val_loss


# error_per_target_var(model, means, stds)
min_loss = get_val_loss_cross_attention_weightless(means, stds)
# Example of processing each chunk

patience = 0
epoch = 0
num_epochs = 5

reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
batches = reader.next_batches(100)

# batches = [pl.read_csv(train_file, n_rows=100000)]


def eval_cross_attention_weightless(min_std, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, means, stds):
    model.eval()
    mask = stds < (1.1 * min_std)

    with torch.no_grad():
        all_preds = np.zeros((1000000, 368))
        all_targets = np.zeros((1000000, 368))
        val_time_start = time.time()
        start_idx = 0

        log_mask = np.zeros(368, dtype=bool)
        log_mask[logs] = True

        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds[:, mask] = 0

            val_preds = val_preds.cpu().numpy()
            val_preds = (val_preds * stds + means)

            tgt = tgt.cpu().numpy()
            tgt = (tgt * stds + means)

            val_preds[:, log_mask] = np.exp(val_preds[:, log_mask]) - np_log_shifts[log_mask]
            tgt[:, log_mask] = np.exp(tgt[:, log_mask]) - np_log_shifts[log_mask]

            batch_size = tgt.shape[0]
            all_targets[start_idx: start_idx + batch_size] = tgt * TARGET_WEIGHTS
            all_preds[start_idx: start_idx + batch_size] = val_preds * TARGET_WEIGHTS
            start_idx += batch_size

        # Calculate R² score
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        avg_val_loss = ss_res / ss_tot

        val_time_end = time.time()

        if avg_val_loss < min_loss:
            print("epoch:", epoch, "model saved", "chunk:", counter, "iterations:", iterations, "val loss:",
                  avg_val_loss, "time:", val_time_end - val_time_start)
            torch.save(model, f"models/{model_name}")
            min_loss = avg_val_loss
            patience = 0
        else:
            print("epoch:", epoch, f"No improvement in validation loss for {patience} chunks.", "chunk:", counter,
                  "iterations:", iterations, "val loss:", avg_val_loss, "time:", val_time_end - val_time_start)
            patience += 1

    return patience, min_loss


print("batches:", len(batches), "shapes:", [batch.shape for batch in batches])

criterion = nn.MSELoss()  # Using MSE for regression


while True:
    if epoch == 0:
        counter = start_from
    else:
        counter = 0

    iterations = 0
    for idx, df in enumerate(batches):
        if idx < counter:
            continue

        prep_chunk_time_start = time.time()

        train_dataset, _, stds, means, shifts = seq2scalar_custom_norm_weightless(min_std, df, FEAT_COLS, TARGET_COLS, transforms, seq_variables_x, scalar_variables_x)

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
        mask = stds < (1.1 * min_std)

        model.train()

        torch_log_mask = torch.zeros(368).bool().cuda()
        torch_log_mask[logs] = True

        for batch_idx, (src, tgt) in enumerate(train_loader):

            optimizer.zero_grad()
            src_2 = get_feature_data(src, variable_order_indices)
            preds = model(src, src_2)
            preds[:, mask] *= 0

            preds = preds * torch_std + torch_mean
            tgt = tgt * torch_std + torch_mean

            preds[:, logs] = torch.exp(preds[:, logs]) - torch_log_shifts[logs]
            tgt[:, logs] = torch.exp(tgt[:, logs]) - torch_log_shifts[logs]

            preds = preds * Targets_norm
            tgt = tgt * Targets_norm

            ss_res = torch.sum((tgt - preds) ** 2)
            ss_tot = torch.sum((tgt - torch.mean(tgt)) ** 2)
            loss = ss_res / ss_tot

            #loss = criterion(preds, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()  # Update learning rate

            total_loss += loss.item()

            iterations += 1
            steps += 1

            if (batch_idx + 1) % 100 == 0:
                train_time_end = time.time()
                print("epoch:", epoch, f', chunk: {counter}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}', "iterations:", iterations, "time:", train_time_end - train_time_start)
                total_loss = 0  # Reset the loss for the next steps
                steps = 0  # Reset step count

        if counter % 10 == 0:
            #error_per_target_var(model)
            patience, min_loss = eval_cross_attention_weightless(min_std, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, means, stds)

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, end of chunk {idx}, Learning Rate: {param_group['lr']}")

        print()
        counter += 1

    epoch += 1
