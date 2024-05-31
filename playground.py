import numpy as np


import polars as pl
import numpy as np

from constants import TARGET_WEIGHTS


# Function to calculate mean and sum of squares in chunks
def calculate_mean_std_in_chunks(file_path, chunk_size, target_cols, weights):
    # Initialize accumulators
    total_count = 0
    mean_accum = np.zeros(num_columns)
    ss_accum = np.zeros(num_columns)  # Sum of squares accumulator

    reader = pl.read_csv_batched(file_path, batch_size=chunk_size)
    batches = reader.next_batches(20)

    for df in batches:
        for col in target_cols:
            y = df.select(target_cols).with_columns(pl.col(col).cast(pl.Float64))

        y_np = y.to_numpy()
        mean = np.mean(y_np, axis=0)
        batch_sum_sq = np.sum((y_np - mean) ** 2, axis=0)

        mean_accum += (mean * df.shape[0])
        ss_accum += batch_sum_sq

        total_count += df.shape[0]

    # Calculate overall mean
    overall_mean = mean_accum / total_count

    # Calculate standard deviation
    overall_var = ss_accum / total_count
    overall_std = np.sqrt(overall_var)

    return overall_mean, overall_std


# Usage example
file_path = 'data/train.csv'
df_header = pl.read_csv(file_path, has_header=True, skip_rows=0, n_rows=1)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]
weights = np.array(TARGET_WEIGHTS)

chunk_size = 500000  # Define the size of each batch
num_columns = 368  # Number of columns in the CSV
num_columns = 556

mean_y, std_y = calculate_mean_std_in_chunks(file_path, chunk_size, FEAT_COLS, TARGET_WEIGHTS)

print("mean:", mean_y.shape, "std:", std_y.shape)
# Save the mean and standard deviation to npy files
np.save('data/mean_x.npy', mean_y)
np.save('data/std_x.npy', std_y)

print("Mean and standard deviation have been saved to mean.npy and std.npy")


for i in range(std_y.shape[0]):
    if std_y[i] < 1e-6:
        print(i, std_y[i])
    else:
        print("=====")

std_y = np.clip(std_y, a_min=1e-8, a_max=None)
