import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from torch.utils.data import DataLoader
from decimal import Decimal

#from cross_attention import CrossAttentionModel
from modified_seq_to_scalar_positional import ModifiedSequenceToScalarTransformer_positional
from neural_net.utils import r2_score
from seq2seq_utils import seq2scalar_32, count_parameters, collate_fn, seq2scalar_custom_norm, \
    seq2scalar_custom_norm_weightless, get_feature_data, get_val_loss_cross_attention_weightless, \
    get_val_loss_cross_attention_weightless_2
from constants import seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length, \
    input_variable_order, TARGET_WEIGHTS

from transformers import get_linear_schedule_with_warmup

train_file = '../data/train_set_2.csv'
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

    transformation_type = transforms[t][0]
    shift = transforms[t][2]

    index = TARGET_COLS.index(t)
    if transformation_type == "log":
        log_shifts.append(shift)
        logs.append(index)
    else:
        log_shifts.append(0)

    #print("Target", index, t, transforms[t])


print("logs:", logs)


# Create a lazy dataframe from the CSV file
lazy_df = pl.scan_csv("../data/validation_set_2.csv")
#
# validation_size = 100000
# Count the rows using a lazy operation
validation_size = 1091032
print(validation_size)

# validation_size = pl.read_csv().shape[0]

# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

chunk_size = 100000  # Define the size of each batch

start_from = 0
if start_from > 0:
    print("WARNING: Starting from chunk:", start_from)

min_std = 1e-12

feature_dim = 25
d_model = 256
nhead = 8
num_encoder_layers = 9
dim_feedforward = 256
output_dim = 368
dropout = 0.1
LEARNING_RATE = 1.2e-4
BATCH_SIZE = 256
model = ModifiedSequenceToScalarTransformer_positional(feature_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_length).cuda()

model_name = f'simple_encoder_{min_std}_nhead_{nhead}_enc_l_{num_encoder_layers}_d_{d_model}_custom_norm_weightless_weighted_train_set_2.model'
# model = torch.load(f"models/{model_name}")
#
# model_name = f'cross_attentional_{min_std}_nhead_{nhead}_enc_l_{num_encoder_layers}_d_{d_model}_custom_norm_weightless_weighted_train_new.model'


print(f'The model has {count_parameters(model):,} trainable parameters')
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

num_warmup_steps = 100
num_training_steps = 400000

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


# Normalize features
stds = []
means = []
shifts = []
for i, col in enumerate(TARGET_COLS):
    transformation_type, (mean, std), shift = transforms[col]
    shifts.append(shift)
    if std < (1.1 * min_std):
        std = min_std
    stds.append(std)
    means.append(mean)


# Convert the lists into numpy arrays
means = np.array(means)
stds = np.array(stds)

Targets_norm = torch.tensor(TARGET_WEIGHTS, dtype=torch.float32).cuda()
torch_means = torch.tensor(means, dtype=torch.float64)
torch_stds = torch.tensor(stds, dtype=torch.float64)
torch_log_shifts = torch.tensor(log_shifts, dtype=torch.float64)

np_log_shifts = np.array(log_shifts)


def memory_eff_eval_2():
    val_file = "../data/validation_set_2.csv"
    val_reader = pl.read_csv_batched(val_file, batch_size=chunk_size)

    # all_preds = np.zeros((validation_size, len(TARGET_COLS)))
    # all_targets = np.zeros((validation_size, len(TARGET_COLS)))
    all_preds = []
    all_targets = []

    start_idx = 0
    sum_rows = 0
    while True:
        try:
            val_df = val_reader.next_batches(1)[0]
            sum_rows += val_df.shape[0]

            if val_df is None:
                break  # No more data to read
        except:
            break

        val_dataset, _ = seq2scalar_custom_norm_weightless(min_std, val_df, FEAT_COLS, TARGET_COLS, transforms,
                                                           seq_variables_x, scalar_variables_x)

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                )

        all_preds, all_targets, start_idx = get_val_loss_cross_attention_weightless_2(model, min_std, logs, torch_log_shifts, all_preds, all_targets, start_idx, torch_means, torch_stds, val_loader)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_preds *= Targets_norm.cpu()
    all_targets *= Targets_norm.cpu()

    loss = torch.mean((all_preds - all_targets)**2)
    #ss_res = torch.sum((all_targets - all_preds) ** 2)

    # ss_tot = torch.sum((all_targets - torch.mean(all_targets, dim=0)) ** 2)
    # loss = ss_res / ss_tot

    # mean_ss_res_vector = np.mean(ss_res_vector, axis=0)
    # mean_ss_tot_vector = np.mean(ss_tot_vector, axis=0)

    # for i in range(len(TARGET_COLS)):
    #     print(transformations[TARGET_COLS[i]], TARGET_WEIGHTS[i], TARGET_COLS[i], mean_ss_res_vector[i], mean_ss_tot_vector[i], mean_all_targets[i], std_all_targets[i])

    return loss.item()


def memory_eff_eval():
    val_file = "../data/validation_set_2.csv"
    val_reader = pl.read_csv_batched(val_file, batch_size=chunk_size//2)

    all_preds = np.zeros((validation_size, 368))
    all_targets = np.zeros((validation_size, 368))
    start_idx = 0
    sum_rows = 0
    while True:
        try:
            val_df = val_reader.next_batches(1)[0]
            sum_rows += val_df.shape[0]

            if val_df is None:
                break  # No more data to read
        except:
            break

        val_dataset, _ = seq2scalar_custom_norm_weightless(min_std, val_df, FEAT_COLS, TARGET_COLS, transforms,
                                                           seq_variables_x, scalar_variables_x)

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                )

        all_preds, all_targets, start_idx = get_val_loss_cross_attention_weightless(model, min_std, logs, np_log_shifts, all_preds, all_targets, start_idx, means, stds, val_loader)

    all_preds *= TARGET_WEIGHTS
    all_targets *= TARGET_WEIGHTS

    mean_all_targets = np.mean(all_targets, axis=0)

    ss_res_vector = (all_targets - all_preds) ** 2
    ss_tot_vector = (all_targets - mean_all_targets) ** 2

    ss_res = np.sum(ss_res_vector)
    ss_tot = np.sum(ss_tot_vector)

    print("ss_res:", ss_res, "ss_tot:", ss_tot)

    loss = ss_res / ss_tot

    # mean_ss_res_vector = np.mean(ss_res_vector, axis=0)
    # mean_ss_tot_vector = np.mean(ss_tot_vector, axis=0)

    # for i in range(len(TARGET_COLS)):
    #     print(transformations[TARGET_COLS[i]], TARGET_WEIGHTS[i], TARGET_COLS[i], mean_ss_res_vector[i], mean_ss_tot_vector[i], mean_all_targets[i], std_all_targets[i])

    return loss


start_eval = time.time()
min_loss = memory_eff_eval_2() #memory_eff_eval()
end_eval = time.time()
print("Initial validation loss:", min_loss, "time:", end_eval - start_eval)
time.sleep(1)

# error_per_target_var(model, means, stds)

# Example of processing each chunk

patience = 0
epoch = 0

# batches = [pl.read_csv(train_file, n_rows=100000)]


def eval_cross_attention_weightless(model, min_loss, patience, epoch, counter, iterations, model_name):
    model.eval()

    val_time_start = time.time()
    loss = memory_eff_eval_2()
    val_time_end = time.time()

    if loss < min_loss:
        print("epoch:", epoch, "model saved", "chunk:", counter, "iterations:", iterations, "val loss:",
              loss, "time:", val_time_end - val_time_start)
        torch.save(model, f"models/{model_name}")
        min_loss = loss
        patience = 0
    else:
        print("epoch:", epoch, f"No improvement in validation loss for {patience} chunks.", "chunk:", counter,
              "iterations:", iterations, "val loss:", loss, "time:", val_time_end - val_time_start)
        patience += 1

    return patience, min_loss


criterion = nn.MSELoss()  # Using MSE for regression


iterations = 0
while True:
    reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
    counter = 0

    while True:

        prep_chunk_time_start = time.time()
        try:
            df = reader.next_batches(1)[0]
            if df is None:
                break  # No more data to read
        except:
            break

        train_dataset, _ = seq2scalar_custom_norm_weightless(min_std, df, FEAT_COLS, TARGET_COLS, transforms, seq_variables_x, scalar_variables_x)

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

        model.train()

        torch_log_mask = torch.zeros(368).bool().cuda()
        torch_log_mask[logs] = True

        for batch_idx, (src, tgt) in enumerate(train_loader):

            optimizer.zero_grad()
            preds = model(src)
            #preds[:, mask] *= 0

            preds = preds * Targets_norm
            tgt = tgt * Targets_norm

            # ss_res = torch.sum((tgt - preds) ** 2)
            # ss_tot = torch.sum((tgt - torch.mean(tgt, dim=0)) ** 2)
            # loss = ss_res / ss_tot

            loss = criterion(preds, tgt)
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
                time.sleep(2)

        if counter % 10 == 0:
            patience, min_loss = eval_cross_attention_weightless(model, min_loss, patience, epoch, counter, iterations, model_name)
            time.sleep(10)

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, end of chunk {counter}, Learning Rate: {param_group['lr']}")

        print()
        counter += 1

    epoch += 1
