import numpy as np
import polars as pl


def calc_mean_after_norm():
    train_file = "../data/train.csv"
    chunk_size = 100000

    df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)

    TARGET_COLS = df_header.columns[557:]

    mean_y = np.load("../data/means_y.npy")
    std_y = np.load("../data/std_y.npy")

    min_std = 1e-12
    std_y = np.clip(std_y, a_min=min_std, a_max=None)

    sums_cols = np.zeros(len(TARGET_COLS), dtype=np.float32)
    sum_rows = 0

    reader = pl.read_csv_batched(train_file, batch_size=chunk_size)

    while True:

        try:
            df = reader.next_batches(1)[0]
            if df is None:
                break  # No more data to read
        except:
            break

        for col in TARGET_COLS:
            y = df.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float32))

        y = y.to_numpy()
        y = (y - mean_y) / std_y
        print(y.shape)

        sums_cols += np.sum(y, axis=0)
        print(list(sums_cols))
        sum_rows += df.shape[0]

    means = sums_cols / sum_rows
    print("means shape:", means.shape)
    np.save('../data/means_y_after_norm.npy', means)

    return means


def calc_r2_denom(means):
    train_file = "../data/train.csv"
    chunk_size = 100000

    df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)

    TARGET_COLS = df_header.columns[557:]

    mean_y = np.load("../data/means_y.npy")
    std_y = np.load("../data/std_y.npy")

    print("mean y:", mean_y.shape)

    min_std = 1e-12
    std_y = np.clip(std_y, a_min=min_std, a_max=None)

    sums_cols = np.zeros(len(TARGET_COLS), dtype=np.float64)
    sum_rows = 0

    reader = pl.read_csv_batched(train_file, batch_size=chunk_size)

    while True:

        try:
            df = reader.next_batches(1)[0]
            if df is None:
                break  # No more data to read
        except:
            break

        for col in TARGET_COLS:
            y = df.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float32))

        y = y.to_numpy()

        y = (y - mean_y) / std_y
        print(y.shape)

        res = (y - means) ** 2
        sums_cols += np.sum(res, axis=0)
        print(list(sums_cols))
        sum_rows += df.shape[0]

    r2_denom = sums_cols / sum_rows
    print("means shape:", r2_denom.shape)
    np.save('../data/r2_denominator.npy', r2_denom)

    return r2_denom


train_file = "../data/train.csv"
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
TARGET_COLS = df_header.columns[557:]

means = calc_mean_after_norm()
for idx, col in enumerate(TARGET_COLS):
    print(idx, col, means[idx])

calc_r2_denom(means)
