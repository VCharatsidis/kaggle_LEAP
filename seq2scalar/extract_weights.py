import polars as pl
import numpy as np

train_file = "../data/sample_submission.csv"
sub = pl.read_csv(train_file, n_rows=100)
print(sub.columns)

first_row = sub.row(0)

print(list(first_row))
