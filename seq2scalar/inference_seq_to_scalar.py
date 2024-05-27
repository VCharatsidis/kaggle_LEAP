import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from seq2scalar.nn_architecture.transformer import TransformerSeq2Seq
from seq2seq_utils import seq2scalar_32, count_parameters, eval_model, to_tensor
from neural_net.utils import r2_score
import polars as pl
from constants import BATCH_SIZE, LEARNING_RATE, seq_variables_x, \
    scalar_variables_x, seq_variables_y, scalar_variables_y, min_std
from transformer_constants import input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, \
    dim_feedforward, dropout

train_file = '../data/train.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=0)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

#calc_x_mean_and_std(file='../data/train.csv', FEAT_COLS=FEAT_COLS)

mean_y = np.load('../data/mean_y.npy')
std_y = np.load('../data/std_y.npy')

mean_x = np.load('../data/mean_x.npy')
std_x = np.load('../data/std_x.npy')

print("mean_y:", mean_y.shape, "std_y:", std_y.shape)

# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

model_name = 'seq2scalar.model'
model = torch.load(model_name)


patience = 0
rounds = 0
num_epochs = 1

test_file = '../data/test.csv'
df = pl.read_csv(test_file)

for col in FEAT_COLS:
    X = df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float64))

# Normalize features
X = X.with_columns([(pl.col(col) - mean_x[i]) / std_x[i] for i, col in enumerate(FEAT_COLS)])

# Reshape the features into the desired shape [batch_size, 60, 25]
batch_size = X.shape[0]
sequence_length = 60

# Convert the numpy array to a PyTorch tensor
tensor_data = to_tensor(X, batch_size, sequence_length, seq_variables_x, scalar_variables_x)
print("test data:", tensor_data.shape)

# tensor_data = tensor_data[:25000]


def collect_predictions_in_batches(model, input_data_test, batch_size):
    start_time = time.time()
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    with torch.no_grad():
        num_samples = input_data_test.size(0)
        for i in range(0, num_samples, batch_size):
            print(i)
            # Select batch of samples
            batch = input_data_test[i:i+batch_size]
            preds = model(batch)
            all_predictions.append(preds.cpu().numpy())

    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    print("Time taken to make predictions:", time.time() - start_time)
    return all_predictions


test_preds = collect_predictions_in_batches(model, tensor_data, batch_size=25000)
print("test predictions shape:", test_preds.shape)

# override constant columns
for i in range(std_y.shape[0]):
    if std_y[i] < min_std * 1.1:
        test_preds[:, i] = 0

test_preds = test_preds * std_y + mean_y
print("test predictions shape:", test_preds.shape)

sub = pd.read_csv("../data/sample_submission.csv")
print(sub.columns.to_list())
print(len(sub.columns.to_list()))

print(sub.iloc[:, 1:].shape)
sub.iloc[:, 1:] = sub.iloc[:, 1:].values * test_preds

test_polars = pl.from_pandas(sub[["sample_id"] + TARGET_COLS])
print(test_polars.shape)
test_polars.write_csv("seq_to_scalar_sub.csv")
print("inference done!")
