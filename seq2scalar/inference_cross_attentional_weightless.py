import pickle
import time

import pandas as pd
import polars as pl
import numpy as np
import torch

from constants import seq_variables_x, scalar_variables_x, TARGET_WEIGHTS
from seq2seq_utils import to_tensor, get_feature_data, variable_order_indices

# Define the path to your pickle file
pickle_file_path = 'transforms_weightless.pickle'

# Load the pickle file
with open(pickle_file_path, 'rb') as handle:
    transforms = pickle.load(handle)

model_name = "models/cross_attentional_1e-12_nhead_8_enc_l_6_d_256_custom_norm_weightless_weighted_train.model"
model = torch.load(model_name)
model.eval()

train_file = '../data/train.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=0)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

log_shifts = {}
inverse_log_shifts = {}
logs = []
inverse_log = []
stds = []
means = []
shifts = []
for t in TARGET_COLS:
    if t not in transforms.keys():
        continue

    type, (mean, std), shift = transforms[t]

    shifts.append(shift)
    stds.append(std)
    means.append(mean)

    index = TARGET_COLS.index(t)
    if type == "log":
        log_shifts[index] = shift
        logs.append(index)

    print("Target", index, t, transforms[t])

stds = np.array(stds)
means = np.array(means)


test_file = '../data/test.csv'
df = pl.read_csv(test_file)
min_std = 1e-12

# Preprocess the features and target columns
for col in FEAT_COLS:
    X = df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float64))

# Normalize features
for i, col in enumerate(FEAT_COLS):
    type, (mean, std), shift = transforms[col]
    if type == 'none':
        continue

    if std < 1.1 * min_std:
        std = min_std

    if type == "log":
        X = X.with_columns((((pl.col(col) + shift).log() - mean) / std))
    else:
        X = X.with_columns(((pl.col(col) - mean) / std))

# Reshape the features into the desired shape [batch_size, 60, 25]
batch_size = X.shape[0]
sequence_length = 60

# Convert the numpy array to a PyTorch tensor
tensor_data = to_tensor(X, batch_size, sequence_length, seq_variables_x, scalar_variables_x)
print("tensor_data input shape:", tensor_data.shape)


def collect_predictions_in_batches(model, input_data_test, batch_size):
    start_time = time.time()
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    mask = stds < (1.1 * min_std)

    with torch.no_grad():
        num_samples = input_data_test.size(0)
        for i in range(0, num_samples, batch_size):
            print(i)
            # Select batch of samples
            batch = input_data_test[i:i + batch_size]
            src_2 = get_feature_data(batch, variable_order_indices)
            val_preds = model(batch, src_2)

            val_preds[:, mask] = 0

            val_preds = val_preds.cpu().numpy()
            val_preds = (val_preds * stds + means)

            for i in range(val_preds.shape[1]):
                if i in logs:
                    val_preds[:, i] = np.exp(val_preds[:, i]) - log_shifts[i]

            val_preds = val_preds * TARGET_WEIGHTS

            all_predictions.append(val_preds)

    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    print("Time taken to make predictions:", time.time() - start_time)
    return all_predictions


test_preds = collect_predictions_in_batches(model, tensor_data, batch_size=25000)
print("test predictions shape:", test_preds.shape)


sub = pd.read_csv("../data/sample_submission.csv")
print(sub.columns.to_list())
print(len(sub.columns.to_list()))

print(sub.iloc[:, 1:].shape)
sub.iloc[:, 1:] = test_preds

test_polars = pl.from_pandas(sub[["sample_id"] + TARGET_COLS])

# REPLACEMENT COLUMNS

REPLACE_FROM = ['ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_20', 'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24', 'ptend_q0002_25', 'ptend_q0002_26']
REPLACE_TO = ['state_q0002_0', 'state_q0002_1', 'state_q0002_2', 'state_q0002_3', 'state_q0002_4', 'state_q0002_5', 'state_q0002_6', 'state_q0002_7', 'state_q0002_8', 'state_q0002_9', 'state_q0002_10', 'state_q0002_11', 'state_q0002_12', 'state_q0002_13', 'state_q0002_14', 'state_q0002_15', 'state_q0002_16', 'state_q0002_17', 'state_q0002_18', 'state_q0002_19', 'state_q0002_20', 'state_q0002_21', 'state_q0002_22', 'state_q0002_23', 'state_q0002_24', 'state_q0002_25', 'state_q0002_26']


print(test_polars.shape)
test_polars.write_csv("cross_seq_to_scalar_weightless.csv")
print("inference done!")

