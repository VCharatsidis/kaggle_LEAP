import polars as pl
import numpy as np

chunk_size = 2000000

train_file = "../data/train.csv"

df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]


def calc_means(ALL_COLS):
    reader = pl.read_csv_batched(train_file, batch_size=chunk_size)

    sums_cols = np.zeros(len(ALL_COLS), dtype=np.float64)
    sum_rows = 0
    while True:

        try:
            df = reader.next_batches(1)[0]
            print(df.shape)
            df = df.select(ALL_COLS)
            # Convert to float64 before summation
            df = df.with_columns([pl.col(col).cast(pl.Float64) for col in ALL_COLS])

            # Calculate the sum for each column
            for i, col in enumerate(ALL_COLS):
                col_sum = df[col].sum()
                sums_cols[i] += col_sum

            sum_rows += df.shape[0]
        except:
            break

    means = sums_cols / sum_rows
    print("means shape:", means.shape)
    return means


def calc_variances(ALL_COLS, means):
    reader = pl.read_csv_batched(train_file, batch_size=chunk_size)

    sums_vars = np.zeros(len(ALL_COLS), dtype=np.float64)
    sum_rows = 0
    while True:

        try:
            df = reader.next_batches(1)[0]
            print(df.shape)
            df = df.select(ALL_COLS)
            # Convert to float64 before summation
            df = df.with_columns([pl.col(col).cast(pl.Float64) for col in ALL_COLS])

            # Calculate the sum for each column
            for i, col in enumerate(ALL_COLS):
                var_column = ((df[col] - means[i]) ** 2).sum()
                sums_vars[i] += var_column

            sum_rows += df.shape[0]
        except:
            break

    vars = sums_vars / sum_rows

    return vars


means_y = calc_means(TARGET_COLS)
np.save('means_y.npy', means_y)

vars_y = calc_variances(TARGET_COLS, means_y)
std_y = np.sqrt(vars_y)
np.save('std_y.npy', std_y)


means_x = calc_means(FEAT_COLS)
np.save('means_x.npy', means_x)

vars_x = calc_variances(FEAT_COLS, means_x)
std_x = np.sqrt(vars_x)
np.save('std_x.npy', std_x)





