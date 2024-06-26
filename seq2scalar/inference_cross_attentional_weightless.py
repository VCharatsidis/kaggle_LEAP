import pickle
import time

import pandas as pd
import polars as pl
import numpy as np
import torch

from constants import seq_variables_x, scalar_variables_x, TARGET_WEIGHTS
from seq2seq_utils import to_tensor, get_feature_data

# Define the path to your pickle file
pickle_file_path = 'transforms_weightless.pickle'

# Load the pickle file
with open(pickle_file_path, 'rb') as handle:
    transforms = pickle.load(handle)

model_name = "models/cross_attentional_1e-12_nhead_8_enc_l_6_d_256_custom_norm_weightless_weighted_train_new.model"
model_name = "models/simple_encoder_1e-12_nhead_8_enc_l_9_d_256_custom_norm_weightless_weighted_train_set_2.model"
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

min_std = 1e-51


def collect_predictions_in_batches(model):
    start_time = time.time()
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    mask = stds < (1.1 * min_std)
    chunk_size = 5000
    counter = 0

    with torch.no_grad():

        val_file = "../data/test.csv"
        val_reader = pl.read_csv_batched(val_file, batch_size=chunk_size)

        while True:
            try:
                val_df = val_reader.next_batches(1)[0]

                if val_df is None:
                    break  # No more data to read
            except:
                break

            # Preprocess the features and target columns
            for col in FEAT_COLS:
                X = val_df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float64))

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
            num_samples = tensor_data.shape[0]
            for i in range(0, num_samples, batch_size):
                print(counter)
                counter += 1
                # Select batch of samples
                batch = tensor_data[i:i + batch_size]
                val_preds = model(batch)

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


test_preds = collect_predictions_in_batches(model)
print("test predictions shape:", test_preds.shape)

df_p_test = pd.DataFrame(test_preds, columns=TARGET_COLS)

df_test = pd.read_csv(test_file)
for idx in range(12, 15):
    df_p_test[f"ptend_q0002_{idx}"] = -df_test[f"state_q0002_{idx}"].to_numpy() / 1200

test_preds = df_p_test.values

sub = pd.read_csv("../data/sample_submission.csv")
print(sub.columns.to_list())
print(len(sub.columns.to_list()))

print(sub.iloc[:, 1:].shape)
sub.iloc[:, 1:] = test_preds

test_polars = pl.from_pandas(sub[["sample_id"] + TARGET_COLS])

# REPLACEMENT COLUMNS

print(test_polars.shape)
test_polars.write_csv("submissions/simple_seq_to_scalar.csv")
print("inference done!")

