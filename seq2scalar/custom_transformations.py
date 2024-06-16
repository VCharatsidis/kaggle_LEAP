# Log transformation and its inverse
import numpy as np
import polars as pl

from constants import TARGET_WEIGHTS

# Load data from a CSV file
df = pl.read_csv('../data/train.csv')

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
    print(f"Number of NaNs: {nan_count}", column)
    assert nan_count == 0, "Data contains NaNs."

    inf_count = np.isinf(data).sum()
    print(f"Number of infinities in {column}: {inf_count}")
    assert inf_count == 0, f"Data contains infinities in column {column}."


# Define and apply the log transformation directly using Polars
def apply_transforms(df):
    transforms = {}
    for column in all_cols:
        skew = df.select(pl.col(column).skew()).to_numpy()[0][0]  # Compute skewness
        print(column, "skey:", skew)

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

        if skew > 1:
            min_value = df.select(pl.col(column).min()).to_numpy()[0][0]
            if min_value <= 0:
                shift = abs(min_value) + 1  # Shift all values to be at least 1
            else:
                shift = 1  # No shift needed if all values are already positive

            # Handling positive skewness
            min_value_after_shift = df.select((pl.col(column) + shift).min()).to_numpy()[0][0]
            assert min_value_after_shift > 0, f"Shift was not sufficient to avoid negative values. min val {min_value_after_shift}, {shift}"
            df = df.with_columns(
                (pl.col(column) + shift).log().alias(column)
            )
            transforms[column] = ('log', shift)

            data = df[column].to_numpy()
            # Check for NaNs
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                print(f"Number of NaNs: {nan_count}", column)
            assert nan_count == 0, "Data contains NaNs."
        # elif skew < -1:
        #     max_value = df.select(pl.col(column).max()).to_numpy()[0][0]
        #
        #     df = df.with_columns(
        #         (max_value - pl.col(column) + 1).log().alias(column)
        #     )
        #     transforms[column] = ('negative_log', max_value)
        #
        #     data = df[column].to_numpy()
        #     # Check for NaNs
        #     nan_count = np.isnan(data).sum()
        #     if nan_count > 0:
        #         print(f"Number of NaNs: {nan_count}", column)
        #     assert nan_count == 0, "Data contains NaNs."


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
with open('transforms_weightless.pickle', 'wb') as handle:
    pickle.dump(transforms, handle, protocol=pickle.HIGHEST_PROTOCOL)
