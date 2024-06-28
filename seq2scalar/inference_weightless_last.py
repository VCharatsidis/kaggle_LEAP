import pickle
import time

import pandas as pd
import polars as pl
import numpy as np
import torch

from constants import seq_variables_x, scalar_variables_x, OFFICIAL_TARGET_WEIGHTS
from seq2seq_utils import to_tensor, get_feature_data

mean_x = np.load("../data/means_x.npy")
std_x = np.load("../data/std_x.npy")

mean_y = np.load("../data/means_y.npy")
std_y = np.load("../data/std_y.npy")

min_std = 1e-12
std_y = np.clip(std_y, a_min=min_std, a_max=None)
std_x = np.clip(std_x, a_min=min_std, a_max=None)

model_name = "models/cross_attention_1e-12_nhead_8_enc_l_9_d_256_weightless_train_set_2.model"
model = torch.load(model_name)
model.eval()

train_file = '../data/train.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=0)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]


test_file = '../data/test.csv'


def collect_predictions_in_batches(model):
    start_time = time.time()
    model.eval()  # Set model to evaluation mode
    all_predictions = []

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
                X = val_df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))

                # Normalize features
            X = X.with_columns([(pl.col(col) - mean_x[i]) / std_x[i] for i, col in enumerate(FEAT_COLS)])

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

                val_preds = val_preds.cpu().numpy()
                val_preds = val_preds * std_y + mean_y
                val_preds = val_preds * OFFICIAL_TARGET_WEIGHTS

                all_predictions.append(val_preds)

    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    print("Time taken to make predictions:", time.time() - start_time)
    return all_predictions


test_preds = collect_predictions_in_batches(model)
print("test predictions shape:", test_preds.shape)

df_p_test = pd.DataFrame(test_preds, columns=TARGET_COLS)

REPLACE_TO = ['ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_20', 'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24', 'ptend_q0002_25', 'ptend_q0002_26']
REPLACE_FROM = ['state_q0002_0', 'state_q0002_1', 'state_q0002_2', 'state_q0002_3', 'state_q0002_4', 'state_q0002_5', 'state_q0002_6', 'state_q0002_7', 'state_q0002_8', 'state_q0002_9', 'state_q0002_10', 'state_q0002_11', 'state_q0002_12', 'state_q0002_13', 'state_q0002_14', 'state_q0002_15', 'state_q0002_16', 'state_q0002_17', 'state_q0002_18', 'state_q0002_19', 'state_q0002_20', 'state_q0002_21', 'state_q0002_22', 'state_q0002_23', 'state_q0002_24', 'state_q0002_25', 'state_q0002_26']

df_test = pd.read_csv(test_file)
for idx in range(0, 27):
    df_p_test[f"ptend_q0002_{idx}"] = -df_test[f"state_q0002_{idx}"].to_numpy() / 1200

test_preds = df_p_test.values * OFFICIAL_TARGET_WEIGHTS

sub = pd.read_csv("../data/sample_submission.csv")
print(sub.columns.to_list())
print(len(sub.columns.to_list()))

print(sub.iloc[:, 1:].shape)
sub.iloc[:, 1:] = test_preds

test_polars = pl.from_pandas(sub[["sample_id"] + TARGET_COLS])

# REPLACEMENT COLUMNS

print(test_polars.shape)
test_polars.write_csv("submissions/cross_weightless_last_no_replacement.csv")
print("inference done!")

