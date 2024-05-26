import torch
import torch.nn as nn

import os
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df = pl.read_csv_batched('data/train.csv')

data = train_df.next_batches(1)[0]
print(data.columns)

FEAT_COLS = data.columns[1:557]
TARGET_COLS = data.columns[557:]

for col in FEAT_COLS:
    X = data.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))
for col in TARGET_COLS:
    y = data.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float32))


X, y = X.to_numpy(), y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

deleted_columns = np.where(np.std(X_train, axis=0) < 1e-6)[0]
X_train_cleaned = np.delete(X_train, deleted_columns, 1)
print(X_train_cleaned.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_cleaned)
X_train_preprocessed = scaler.transform(X_train_cleaned)

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

model = XGBRegressor(tree_method='hist', device="cuda")
multi_model = MultiOutputRegressor(model)

multi_model.fit(X_train_preprocessed, y_train)

X_test_cleaned = np.delete(X_test, deleted_columns, 1)
X_test_preprocessed = scaler.transform(X_test_cleaned)
predictions = multi_model.predict(X_test_preprocessed)

print("predictions:", predictions.shape)

score_test = multi_model.score(X_test_preprocessed, y_test)
score_train = multi_model.score(X_train_preprocessed, y_train)

print("Score Test:", score_test, "Score Train:", score_train)


test_df = pl.read_csv('data/test.csv')

for col in FEAT_COLS:
    test_dataframe = test_df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))

test_dataframe = test_dataframe.to_numpy()

print("test dataframe shape:", test_dataframe.shape)

test_dataframe_cleaned = np.delete(test_dataframe, deleted_columns, 1)
test_dataframe_preprocessed = scaler.transform(test_dataframe_cleaned)
test_preds = multi_model.predict(test_dataframe_preprocessed)

import pandas as pd
sub = pd.read_csv("data/sample_submission.csv")
print(sub.columns.values)
print(sub.shape)
sub.head()
sub.iloc[:, 1:] *= test_preds
sub.head()
print(sub.shape)

test_polars = pl.from_pandas(sub[["sample_id"]+TARGET_COLS])
test_polars.write_csv("submission.csv")

