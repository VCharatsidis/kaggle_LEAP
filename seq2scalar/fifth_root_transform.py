# Log transformation and its inverse
import numpy as np
import polars as pl

from constants import TARGET_WEIGHTS

# Load data from a CSV file
df = pl.read_csv('../data/train.csv', n_rows=1000000)

FEAT_COLS = df.columns[1:557]
TARGET_COLS = df.columns[557:]


all_cols = FEAT_COLS + TARGET_COLS
df = df.with_columns([
    pl.col(col).cast(pl.Float64) for col in all_cols
])

# df = df.with_columns(
#     [pl.col(column) * weight for column, weight in zip(TARGET_COLS, TARGET_WEIGHTS)]
# )


for column in all_cols:
    data = df[column].to_numpy()
    # Check for NaNs
    nan_count = np.isnan(data).sum()
    assert nan_count == 0, "Data contains NaNs."

    inf_count = np.isinf(data).sum()
    assert inf_count == 0, f"Data contains infinities in column {column}."


def custom_eight_root(x):
    return np.sign(x) * np.abs(x) ** (1 / 8)


def custom_fifth_root(x):
    return np.sign(x) * np.abs(x) ** (1 / 5)


def cube_third_root(x):
    return np.sign(x) * np.abs(x) ** (1 / 3)


# Define and apply the log transformation directly using Polars
def apply_transforms(df):
    transforms = {}
    for column in all_cols:

        data = df[column].to_numpy()
        # Check for NaNs
        nan_count = np.isnan(data).sum()
        assert nan_count == 0, "Data contains NaNs."

        # Check for infinities
        inf_count = np.isinf(data).sum()
        assert inf_count == 0, "Data contains infinities."

        # Skip transformation if all values are the same (no variability)
        if np.all(data == data[0]):
            transforms[column] = ('none', None)
            continue

        old_skew = df.select(pl.col(column).skew()).to_numpy()[0][0]  # Compute skewness

        if (old_skew > 1) or (old_skew < -1):
            if (old_skew < -5) or (old_skew > 5):
                df = df.with_columns([
                    pl.col(column).map_elements(custom_eight_root, return_dtype=pl.Float64).alias(column)
                ])
                transforms[column] = ('root_8', None)
            elif (old_skew < -3) or (old_skew > 3):
                df = df.with_columns([
                    pl.col(column).map_elements(custom_fifth_root, return_dtype=pl.Float64).alias(column)
                ])
                transforms[column] = ('root_5', None)
            else:
                df = df.with_columns([
                    pl.col(column).map_elements(cube_third_root, return_dtype=pl.Float64).alias(column)
                ])
                transforms[column] = ('root_3', None)

        new_skew = df.select(pl.col(column).skew()).to_numpy()[0][0]  # Compute skewness
        print(column, old_skew, "skew after cube root:", new_skew)

    return df, transforms


df, transforms = apply_transforms(df)

# Standardization
for column in all_cols:
    if column in transforms.keys():
        print(transforms[column])

    skew = df.select(pl.col(column).skew()).to_numpy()[0][0]  # Compute skewness
    print(column, "skey after log:", skew)

    data = df[column].to_numpy()
    # Check for NaNs
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        print(f"Number of NaNs: {nan_count}", column)
    # assert nan_count == 0, "Data contains NaNs."

    mean, std = df[column].mean(), df[column].std()
    if column in transforms.keys():
        transforms[column] = (transforms[column][0], (mean, std), transforms[column][1])
    else:
        transforms[column] = ('standardize', (mean, std), 0)


for key, value in transforms.items():
    print(key, value)

print(len(transforms))

import pickle
with open('transforms_weightless_root.pickle', 'wb') as handle:
    pickle.dump(transforms, handle, protocol=pickle.HIGHEST_PROTOCOL)
