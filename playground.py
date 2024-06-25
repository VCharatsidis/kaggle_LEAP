import numpy as np
import polars as pl



train_file = 'data/train_set_2.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

from constants import TARGET_WEIGHTS

mean_y = np.load("data/means_y.npy")
for idx, col in enumerate(TARGET_COLS):
    print(idx, col, mean_y[idx])

input()

REPLACE_TO = ['ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_20', 'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24', 'ptend_q0002_25', 'ptend_q0002_26']

my_weights = []
for idx, col in enumerate(TARGET_COLS):
    print(idx, col, TARGET_WEIGHTS[idx])
    if col in REPLACE_TO:
        my_weights.append(0)
    else:
        my_weights.append(TARGET_WEIGHTS[idx])

print(my_weights)
