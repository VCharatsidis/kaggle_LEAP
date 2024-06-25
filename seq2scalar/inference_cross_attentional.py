import time

import numpy as np
import pandas as pd
import torch
from seq2seq_utils import to_tensor
import polars as pl
from constants import seq_variables_x, scalar_variables_x, input_variable_order

train_file = '../data/train.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=0)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

#calc_x_mean_and_std(file='../data/train.csv', FEAT_COLS=FEAT_COLS)

mean_y = np.load('../data/mean_weighted_y.npy')
std_y = np.load('../data/std_weighted_y.npy')

min_std = 1e-12
std_y = np.clip(std_y, a_min=min_std, a_max=None)

mean_x = np.load('../data/mean_x.npy')
std_x = np.load('../data/std_x.npy')
std_x = np.clip(std_x, a_min=min_std, a_max=None)

print("mean_y:", mean_y.shape, "std_y:", std_y.shape)

# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

#model_name = f'models/seq2scalar_weighted_32_positional_{min_std}.model'
model_name = "models/cross_attentional_1e-12_2.model"
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


def get_feature_data(tensor_data, variable_order_indices):
    # Reshape and reorder to N, 25, 60
    tensor_data = tensor_data.permute(0, 2, 1)  # Change to N, 25, 60
    tensor_data_ordered = tensor_data[:, variable_order_indices, :]  # Reorder the second dimension as per the desired order

    # The tensor_data_ordered is now in the shape N, 25, 60 with variables ordered as specified
    return tensor_data_ordered


# Map the desired order of variables to their indices in the original tensor
variable_order_indices = [
    seq_variables_x.index(var) if var in seq_variables_x else len(seq_variables_x) + scalar_variables_x.index(var)
    for var in input_variable_order]


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
            src_2 = get_feature_data(batch, variable_order_indices)
            preds = model(batch, src_2)
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
sub.iloc[:, 1:] *= test_preds

test_polars = pl.from_pandas(sub[["sample_id"] + TARGET_COLS])

# REPLACEMENT COLUMNS



print(test_polars.shape)
test_polars.write_csv("cross_seq_to_scalar_weighted_sub_2.csv")
print("inference done!")
