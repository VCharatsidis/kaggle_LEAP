import numpy as np
import torch
import polars as pl
import pandas as pd

from constants import min_std

file = '../data/test.csv'

df = pl.read_csv(file)

df_train = pl.read_csv('../data/train.csv', n_rows=10)
column_names = df_train.columns

FEAT_COLS = column_names[1:557]
TARGET_COLS = column_names[557:]

for col in FEAT_COLS:
    X_test = df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))

X_test = X_test.to_numpy()

mean_x = np.load('../data/mean_x.npy')
std_x = np.load('../data/std_x.npy')

X_test = (X_test - mean_x.reshape(1, -1)) / std_x.reshape(1, -1)

# Convert arrays to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32).cuda()

model = torch.load('models/r2_best_model_GLU.model')
model.eval()
with torch.no_grad():
    test_preds = model(X_test).detach().cpu().numpy()
print("test preds shape:", test_preds.shape)

model_2 = torch.load('models/r2_best_model_GLU_2.model')
model_2.eval()
with torch.no_grad():
    test_preds_2 = model_2(X_test).detach().cpu().numpy()
print("test preds 2 shape:", test_preds_2.shape)

test_preds = (test_preds + test_preds_2) / 2

mean_y = np.load('../data/mean_y.npy')
std_y = np.load('../data/std_y.npy')

# override constant columns
for i in range(std_y.shape[0]):
    if std_y[i] < min_std * 1.1:
        test_preds[:, i] = 0

test_preds = test_preds * std_y.reshape(1, -1) + mean_y.reshape(1, -1)

sub = pd.read_csv("../data/sample_submission.csv")
sub.iloc[:, 1:] *= test_preds

test_polars = pl.from_pandas(sub[["sample_id"] + TARGET_COLS])
print("test_polars:", test_polars.shape)
test_polars.write_csv("r2_nn_submission_GLU_two_models.csv")
print("inference done!")


# Test

# mean_y = np.load('mean_y.npy')
# std_y = np.load('std_y.npy')
# print("mean_y:", mean_y.shape, "std_y:", std_y.shape, "reshaped:", std_y.reshape(1, -1).shape, mean_y.reshape(1, -1).shape)
#
# df = pd.read_csv("../data/test.csv")
#
# model = torch.load('best_model.model')
# model.eval()
#
# xt = df.iloc[:, 1:557].to_numpy().astype(np.float32)
# del df
#
#
# df = pd.read_csv("../data/train.csv", nrows=1)
# y = df.iloc[:, 557:].to_numpy().astype(np.float32)
# output_size = y.shape[1]
#
# predt = np.zeros([xt.shape[0], output_size], dtype=np.float32)  # output_size is the dimension of your model's output
# batch_size = 1024 * 128  # Batch size for inference
#
# i1 = 0
# for i in range(10000):
#     i2 = np.minimum(i1 + batch_size, xt.shape[0])
#     if i1 == i2:  # Break the loop if range does not change
#         break
#
#     # Convert the current slice of xt to a PyTorch tensor
#     inputs = torch.from_numpy(xt[i1:i2, :]).float().cuda()
#
#     # No need to track gradients for inference
#     with torch.no_grad():
#         outputs = model(inputs)  # Get model predictions
#         print(outputs.shape, inputs.shape, predt.shape)
#         predt[i1:i2, :] = outputs.cpu().numpy()  # Store predictions in predt
#
#     print(np.round(i2 / predt.shape[0], 2))  # Print the percentage completion
#     i1 = i2  # Update i1 to the end of the current batch
#
#     if i2 >= xt.shape[0]:
#         break
#
# # override constant columns
# for i in range(std_y.shape[0]):
#     if std_y[i] < min_std * 1.1:
#         predt[:, i] = 0
#
# # undo y scaling
# predt = predt * std_y.reshape(1, -1) + mean_y.reshape(1, -1)
#
# ss = pd.read_csv("../data/sample_submission.csv")
# ss.iloc[:, 1:] *= predt
# ss.to_csv("nn_submission.csv", index=False)
# print("inference done!")



