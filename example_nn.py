import random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time
import warnings

from constants import min_std
from neural_net.nn_architecture.ffnn_example import FFNN
from neural_net.utils import NumpyDataset, r2_score

warnings.filterwarnings('ignore', category=FutureWarning)
t0 = time()
np.random.seed(42)
random.seed(42)

DEBUGGING = False

# load train data
if DEBUGGING:
    n_rows = 10000
else:
    n_rows = 2000000

df = pd.read_csv("data/train.csv", nrows=n_rows)
x = df.iloc[:, 1:557].to_numpy().astype(np.float32)
y = df.iloc[:, 557:].to_numpy().astype(np.float32)


# read test
if not DEBUGGING:
    df = pd.read_csv("data/test.csv")
    xt = df.iloc[:, 1:557].to_numpy().astype(np.float32)
    del df
    gc.collect()


# norm X
mx = x.mean(axis=0)
sx = np.maximum(x.std(axis=0), min_std)
x = (x - mx.reshape(1, -1)) / sx.reshape(1, -1)
if not DEBUGGING:
    xt = (xt - mx.reshape(1, -1)) / sx.reshape(1, -1)

# norm Y
my = y.mean(axis=0)
sy = np.maximum(np.sqrt((y*y).mean(axis=0)), min_std)
y = (y - my.reshape(1, -1)) / sy.reshape(1, -1)

print("mean_y:", my.shape, "std_y:", sy.shape, "reshaped:", sy.reshape(1, -1).shape, my.reshape(1, -1).shape)
input()

dataset = NumpyDataset(x, y)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 4000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size = x.shape[1]
output_size = y.shape[1]
hidden_size = input_size + output_size
model = FFNN(input_size, [3*hidden_size, 2*hidden_size, hidden_size, 2*hidden_size, 3*hidden_size], output_size).cuda()
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.MSELoss()  # Using MSE for regression
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

# Training loop
epochs = 100000
best_val_loss = float('inf')  # Set initial best as infinity
best_model_state = None  # To store the best model's state
patience_count = 0
patience = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    steps = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = r2_score(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        # Print every 10 steps
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}')
            total_loss = 0  # Reset the loss for the next steps
            steps = 0  # Reset step count

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += r2_score(outputs, labels).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')

    scheduler.step(avg_val_loss)  # Adjust learning rate

    # Update best model if current epoch's validation loss is lower
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()  # Save the best model state
        torch.save(model, 'example_r2_nn_best_model.model')
        patience_count = 0
        print("Validation loss decreased, saving new best model and resetting patience counter.")
    else:
        patience_count += 1
        print(f"No improvement in validation loss for {patience_count} epochs.")

    if patience_count >= patience:
        print("Stopping early due to no improvement in validation loss.")
        break


# Test
if not DEBUGGING:
    model.load_state_dict(best_model_state)
    torch.save(model, 'example_r2_nn_best_model.model')
    model.eval()
    predt = np.zeros([xt.shape[0], output_size], dtype=np.float32)  # output_size is the dimension of your model's output
    batch_size = 1024 * 128  # Batch size for inference

    i1 = 0
    for i in range(10000):
        i2 = np.minimum(i1 + batch_size, xt.shape[0])
        if i1 == i2:  # Break the loop if range does not change
            break

        # Convert the current slice of xt to a PyTorch tensor
        inputs = torch.from_numpy(xt[i1:i2, :]).float().cuda()

        # No need to track gradients for inference
        with torch.no_grad():
            outputs = model(inputs)  # Get model predictions
            predt[i1:i2, :] = outputs.cpu().numpy()  # Store predictions in predt

        print(np.round(i2 / predt.shape[0], 2))  # Print the percentage completion
        i1 = i2  # Update i1 to the end of the current batch

        if i2 >= xt.shape[0]:
            break


if not DEBUGGING:
    # submit
    # override constant columns
    for i in range(sy.shape[0]):
        if sy[i] < min_std * 1.1:
            predt[:, i] = 0

    # undo y scaling
    predt = predt * sy.reshape(1, -1) + my.reshape(1, -1)

    ss = pd.read_csv("data/sample_submission.csv")
    ss.iloc[:, 1:] *= predt
    ss.to_csv("nn_r2_example_submission.csv", index=False)
