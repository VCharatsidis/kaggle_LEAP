import numpy as np
import polars as pl

from constants import TARGET_WEIGHTS

file_path = "../data/train.csv"

# Read only the header to get column names
df_header = pl.read_csv(file_path, has_header=True, n_rows=1)
TARGET_COLS = df_header.columns[557:]

# Read the full dataset (or the required portion) into Polars DataFrame
y = pl.read_csv(file_path, columns=TARGET_COLS, n_rows=3000000)

# Convert TARGET_WEIGHTS to a NumPy array if it's not already
weights = np.array(TARGET_WEIGHTS)

# Convert the DataFrame to a NumPy array
y_np = y.to_numpy()

# Apply weights
y_weighted = y_np * weights

# Calculate mean and standard deviation
mean = np.mean(y_weighted, axis=0)
std_dev = np.std(y_weighted, axis=0)

# Print the shapes of the results
print("Mean shape:", mean.shape)
print("Standard Deviation shape:", std_dev.shape)

# Save the mean and standard deviation to npy files
np.save('../data/mean_weighted_y.npy', mean)
np.save('../data/std_weighted_y.npy', std_dev)

print("Mean and standard deviation have been saved to mean_weighted_y.npy and std_weighted_y.npy")
