import polars as pl
import numpy as np

from constants import TARGET_WEIGHTS


# Function to calculate mean and sum of squares in chunks
def calculate_mean_std_in_chunks(file_path, chunk_size, target_cols):
    # Initialize accumulators
    total_count = 0
    mean_accum = np.zeros(num_columns)
    ss_accum = np.zeros(num_columns)  # Sum of squares accumulator

    # Use Polars lazy frame to scan the CSV file
    lazy_df = pl.scan_csv(file_path, new_columns=target_cols)

    # Collect batches in chunks
    batches = lazy_df.collect().to_pandas().groupby(np.arange(len(lazy_df.collect())) // chunk_size)

    for _, chunk in batches:
        chunk = chunk.select(pl.all().cast(pl.Float64))  # Ensure all target columns are float64

        chunk = chunk.to_numpy()
        chunk = chunk * weights

        chunk_count = len(chunk)
        chunk_mean = np.mean(chunk, axis=0)
        chunk_var = np.var(chunk, axis=0, ddof=0)
        chunk_ss = chunk_var * (chunk_count - 1)  # Sum of squares

        mean_accum += chunk_mean * chunk_count
        ss_accum += chunk_ss

        total_count += chunk_count

    # Calculate overall mean
    overall_mean = mean_accum / total_count

    # Calculate overall variance
    overall_var = ss_accum / (total_count - 1)

    # Calculate standard deviation
    overall_std = np.sqrt(overall_var)

    return overall_mean, overall_std


# Usage example
file_path = '../data/train.csv'
df_header = pl.read_csv(file_path, has_header=True, skip_rows=0, n_rows=1)
TARGET_COLS = df_header.columns[557:]
weights = np.array(TARGET_WEIGHTS)

chunk_size = 100000  # Adjust chunk size based on your memory capacity
num_columns = 368  # Number of columns in the CSV

mean, std = calculate_mean_std_in_chunks(file_path, chunk_size, TARGET_COLS)

print("mean:", mean.shape, "std:", std.shape)
# Save the mean and standard deviation to npy files
np.save('mean_weighted_y.npy', mean)
np.save('std_weighted_y.npy', std)

print("Mean and standard deviation have been saved to mean.npy and std.npy")
