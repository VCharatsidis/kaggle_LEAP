import polars as pl

df = pl.read_csv('../data/train.csv')

# Shuffle the DataFrame
df = df.shuffle(seed=42)  # seed for reproducibility

# Calculate split index
split_index = int(0.75 * len(df))

df_train = df[:split_index].write_csv('../data/train_big.csv')
df_test = df[split_index:].write_csv('../data/validation_big.csv')

