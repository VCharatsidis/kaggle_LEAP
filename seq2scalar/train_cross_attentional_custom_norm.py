import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from torch.utils.data import DataLoader

from cross_attention import CrossAttentionModel
from neural_net.utils import r2_score
from seq2seq_utils import seq2scalar_32, count_parameters, collate_fn, seq2scalar_custom_norm
from constants import seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length, \
    input_variable_order, TARGET_WEIGHTS

train_file = '../data/train_set.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

import pickle

# Define the path to your pickle file
pickle_file_path = 'transforms.pickle'

# Load the pickle file
with open(pickle_file_path, 'rb') as handle:
    transforms = pickle.load(handle)

for key, value in transforms.items():
    print(key, value)


min_std = 1e-12


# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

chunk_size = 500000  # Define the size of each batch

model_name = f'cross_attentional_{min_std}_custom_norm.model'

feature_dim = 25
d_model = 128
nhead = 4
num_encoder_layers = 2
dim_feedforward = 256
output_dim = 368
dropout = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
#model = CrossAttentionModel(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()

model = torch.load(f"models/{model_name}")

print(f'The model has {count_parameters(model):,} trainable parameters')
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = 20000  # Total number of training steps
num_warmup_steps = 100     # Number of steps to warm up the learning rate

min_std = 1e-12
val_data = pl.read_csv("../data/validation_set.csv")
val_dataset, _, _ = seq2scalar_custom_norm(min_std, val_data, FEAT_COLS, TARGET_COLS, transforms, seq_variables_x, scalar_variables_x)

log_shifts = {}
logs = []
inverse_log = []
target_stds = []
target_means = []
shifts = []
for idx, t in enumerate(TARGET_COLS):
    type, (mean, std), shift = transforms[t]

    shifts.append(shift)
    target_stds.append(std)
    target_means.append(mean)

    if type == "log":
        log_shifts[idx] = shift
        logs.append(idx)

    print("Target", idx, t, transforms[t])

target_stds = np.array(target_stds)
target_means = np.array(target_means)


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


def error_per_target_var(model):
    model.eval()
    with torch.no_grad():
        val_loss = torch.zeros([len(TARGET_COLS)], dtype=torch.float64).cuda()
        mask = target_stds < (1.1 * min_std)

        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds[:, mask] = 0
            ss_res = (tgt - val_preds) ** 2
            score = torch.sum(ss_res, dim=0)

            val_loss += score

        mean_val = val_loss / val_data.shape[0]

        for i in range(len(TARGET_COLS)):
            print(TARGET_COLS[i], round(mean_val[i].item(), 5), round(TARGET_WEIGHTS[i], 5))

        print(mean_val.mean())


def get_val_loss_cross_attention():
    model.eval()
    mask = target_stds < (1.1 * min_std)
    with torch.no_grad():
        all_preds = []
        all_targets = []
        val_time_start = time.time()
        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds[:, mask] = 0

            val_preds = val_preds.cpu().numpy()
            tgt = tgt.cpu().numpy()

            val_preds = (val_preds * target_stds + target_means)
            tgt = (tgt * target_stds + target_means)

            for i in range(val_preds.shape[1]):
                if i in logs:
                    val_preds[:, i] = np.exp(val_preds[:, i]) - log_shifts[i]
                    tgt[:, i] = np.exp(tgt[:, i]) - log_shifts[i]

            tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
            val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

            all_preds.append(val_preds)
            all_targets.append(tgt)

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate R² score
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        avg_val_loss = ss_res / ss_tot

        val_time_end = time.time()

        print("time:", val_time_end - val_time_start, "val loss:", avg_val_loss)
        return avg_val_loss


min_loss = get_val_loss_cross_attention()
error_per_target_var(model)

# Example of processing each chunk

patience = 0
epoch = 0
num_epochs = 5

reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
batches = reader.next_batches(20)


def eval_cross_attention(min_std, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, stds):
    model.eval()
    mask = target_stds < (1.1 * min_std)
    with torch.no_grad():
        val_time_start = time.time()
        all_preds = []
        all_targets = []
        for src, tgt in val_loader:
            src_2 = get_feature_data(src, variable_order_indices)
            val_preds = model(src, src_2)
            val_preds[:, mask] = 0

            val_preds = val_preds.cpu().numpy()
            val_preds = (val_preds * target_stds + target_means)

            tgt = tgt.cpu().numpy()
            tgt = (tgt * target_stds + target_means)

            for i in range(val_preds.shape[1]):
                if i in logs:
                    val_preds[:, i] = np.exp(val_preds[:, i]) - log_shifts[i]
                    tgt[:, i] = np.exp(tgt[:, i]) - log_shifts[i]

            tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
            val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

            all_preds.append(val_preds)
            all_targets.append(tgt)

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate R² score
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        avg_val_loss = ss_res / ss_tot

        val_time_end = time.time()

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
        counter = 0
    else:
        counter = 0

    iterations = 0
    for idx, df in enumerate(batches):
        if idx < counter:
            continue

        prep_chunk_time_start = time.time()

        train_dataset, _, _ = seq2scalar_custom_norm(min_std, df, FEAT_COLS, TARGET_COLS, transforms, seq_variables_x, scalar_variables_x)

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
        mask = target_stds < (1.1 * min_std)

        model.train()

        for batch_idx, (src, tgt) in enumerate(train_loader):

            optimizer.zero_grad()
            src_2 = get_feature_data(src, variable_order_indices)
            preds = model(src, src_2)
            preds[:, mask] *= 0

            loss = criterion(preds, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

            iterations += 1
            steps += 1

            if (batch_idx + 1) % 100 == 0:
                train_time_end = time.time()
                print("epoch:", epoch, f', chunk: {counter}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}', "iterations:", iterations, "time:", train_time_end - train_time_start)
                total_loss = 0  # Reset the loss for the next steps
                steps = 0  # Reset step count

        #error_per_target_var(model)
        patience, min_loss = eval_cross_attention(min_std, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, target_stds)

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, end of chunk {idx}, Learning Rate: {param_group['lr']}")

        print()
        counter += 1

    epoch += 1
