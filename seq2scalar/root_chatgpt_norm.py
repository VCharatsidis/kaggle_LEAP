import polars as pl
import numpy as np
from scipy.stats import skew
from functools import partial
import pickle

# Define custom root functions
def custom_eight_root(x):
    return np.sign(x) * np.abs(x) ** (1 / 8)

def custom_fifth_root(x):
    return np.sign(x) * np.abs(x) ** (1 / 5)

def cube_third_root(x):
    return np.sign(x) * np.abs(x) ** (1 / 3)

# Define inverse functions
def inverse_custom_root(x, exponent):
    return np.sign(x) * np.abs(x) ** exponent

# Inverse cube root function
def inverse_cube_root(x):
    return x ** 3

# Function to apply normalization based on skewness
def normalize_column(df, column):
    col_skew = df.select(pl.col(column).skew()).to_numpy()[0][0]
    transforms = {}

    print(f"\nColumn: {column}, Initial Skew: {col_skew}")
    print("Initial Values:", df[column].to_list())

    if col_skew > 1 or col_skew < -1:
        if col_skew < -5 or col_skew > 5:
            df = df.with_columns([
                pl.col(column).map_elements(custom_eight_root, return_dtype=pl.Float64).alias(column)
            ])
            transforms[column] = ('root_8', None)
            print(f"Applied custom_eight_root on {column}")
        elif col_skew < -3 or col_skew > 3:
            df = df.with_columns([
                pl.col(column).map_elements(custom_fifth_root, return_dtype=pl.Float64).alias(column)
            ])
            transforms[column] = ('root_5', None)
            print(f"Applied custom_fifth_root on {column}")
        else:
            df = df.with_columns([
                pl.col(column).map_elements(cube_third_root, return_dtype=pl.Float64).alias(column)
            ])
            transforms[column] = ('root_3', None)
            print(f"Applied cube_third_root on {column}")

        new_skew = df.select(pl.col(column).skew()).to_numpy()[0][0]  # Compute skewness
        print(f"New Skew: {new_skew}")
        if new_skew < -1:
            print("Transformed Values:", df[column].to_list())
            input()
    else:
        print(f"No transformation applied on {column}")

    return df, transforms

# Function to invert the transformation
def invert_transformation(df, column, transform_type):
    if transform_type == 'root_8':
        df = df.with_columns([
            pl.col(column).map_elements(partial(inverse_custom_root, exponent=8), return_dtype=pl.Float64).alias(column)
        ])
    elif transform_type == 'root_5':
        df = df.with_columns([
            pl.col(column).map_elements(partial(inverse_custom_root, exponent=5), return_dtype=pl.Float64).alias(column)
        ])
    elif transform_type == 'root_3':
        df = df.with_columns([
            pl.col(column).map_elements(inverse_cube_root, return_dtype=pl.Float64).alias(column)
        ])
    return df

# Load data from a CSV file
df = pl.read_csv('../data/train.csv', n_rows=100)

FEAT_COLS = df.columns[1:557]
TARGET_COLS = df.columns[557:]
all_cols = FEAT_COLS + TARGET_COLS

# Ensure all columns are float64
df = df.with_columns([pl.col(col).cast(pl.Float64) for col in all_cols])

# Validate data for NaNs and infinities
for column in all_cols:
    data = df[column].to_numpy()
    nan_count = np.isnan(data).sum()
    assert nan_count == 0, "Data contains NaNs."
    inf_count = np.isinf(data).sum()
    assert inf_count == 0, f"Data contains infinities in column {column}."

# Apply transformations based on skewness
transforms = {}
for column in all_cols:
    df, column_transforms = normalize_column(df, column)
    transforms.update(column_transforms)

# Print transformed DataFrame and transformation details
print("\nTransformed DataFrame:")
print(df)
print("Transforms applied:", transforms)

# Example inference: Invert transformations
for column, (transform_type, _) in transforms.items():
    df = invert_transformation(df, column, transform_type)

# Print DataFrame after inverse transformation
print("\nDataFrame after Inverse Transformation:")
print(df)

# Save transformation details
with open('transforms_weightless_root.pickle', 'wb') as handle:
    pickle.dump(transforms, handle, protocol=pickle.HIGHEST_PROTOCOL)
