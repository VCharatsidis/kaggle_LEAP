import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from neural_net.nn_architecture.ffnn_example import FFNN, FFNN_gelu
from neural_net.nn_architecture.tabular_mlp import TabularModel
from neural_net.utils import mae, NumpyDataset, get_val_loader, calc_x_mean_and_std, r2_score
import polars as pl
from constants import num_targets, num_features, BATCH_SIZE, LEARNING_RATE, min_std

train_file = '../data/train_set.csv'
# Read only the first row (header) to get column names
df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=0)
FEAT_COLS = df_header.columns[1:557]
TARGET_COLS = df_header.columns[557:]

#calc_x_mean_and_std(file='../data/train.csv', FEAT_COLS=FEAT_COLS)

mean_y = np.load('../data/mean_y.npy')
std_y = np.load('../data/std_y.npy')

mean_x = np.load('../data/mean_x.npy')
std_x = np.load('../data/std_x.npy')

# Number of columns and their names
column_names = df_header.columns
num_columns = len(column_names)
print("num columns:", num_columns)

chunk_size = 1000000  # Define the size of each batch

# Instantiate model

hidden_size = num_features + num_targets

DIM_FEATURES = 556
DIM_TARGETS = 368
model = TabularModel([DIM_FEATURES, 1024, 512, DIM_TARGETS]).cuda()

#model = FFNN_gelu(num_features, [hidden_size, hidden_size, hidden_size, hidden_size], num_targets).cuda()
#model = torch.load('r2_best_model_GLU.model')
print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=6, verbose=False)
scheduler = optim.lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=50)

criterion = nn.MSELoss()  # Using MSE for regression
min_loss = 10000000000000
# Example of processing each chunk

val_loader = get_val_loader('../data/validation_set.csv', FEAT_COLS, TARGET_COLS, mean_y, std_y, mean_x, std_x, batch_size=BATCH_SIZE)

# Set the random seed for PyTorch
#torch.manual_seed(42)  # You can use any seed number here

patience = 0
rounds = 0
num_epochs = 2

reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
batches = reader.next_batches(10)
print("batches:", len(batches), "shapes:", [batch.shape for batch in batches])

while patience < (4 * num_epochs):
    counter = 0
    iterations = 0
    for data in batches:
        prep_chunk_time_start = time.time()
        for col in FEAT_COLS:
            X = data.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))
        for col in TARGET_COLS:
            y = data.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float32))

        X, y = X.to_numpy(), y.to_numpy()
        y = (y - mean_y.reshape(1, -1)) / std_y.reshape(1, -1)

        # norm X
        X = (X - mean_x.reshape(1, -1)) / std_x.reshape(1, -1)

        train_dataset = NumpyDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        prep_chunk_time_end = time.time()

        print("round:", rounds, "chunk:", counter, "prep chunk time:", prep_chunk_time_end - prep_chunk_time_start)

        # Training loop
        model.train()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            steps = 0
            train_time_start = time.time()
            for batch_idx, (features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                iterations += 1

                total_loss += loss.item()
                steps += 1

                # Print every 10 steps
                if (batch_idx + 1) % 100 == 0:
                    train_time_end = time.time()
                    print("round:", rounds, f'Epoch {epoch + 1}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}', "iterations:", iterations, "time:", train_time_end - train_time_start)
                    total_loss = 0  # Reset the loss for the next steps
                    steps = 0  # Reset step count

            if epoch % 2 == 0:
                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_time_start = time.time()
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        val_loss += r2_score(outputs, labels).item()

                    val_time_end = time.time()

                    avg_val_loss = val_loss / len(val_loader)
                    #scheduler.step(avg_val_loss)  # Adjust learning rate

                    if avg_val_loss < min_loss:
                        print("round:", rounds, "model saved", "chunk:", counter, "iterations:", iterations, "val loss:", avg_val_loss, "time:", val_time_end - val_time_start)
                        torch.save(model, 'models/r2_best_model_GLU.model')
                        min_loss = avg_val_loss
                        patience = 0
                    else:
                        print("round:", rounds, f"No improvement in validation loss for {patience} epochs.", "chunk:", counter, "iterations:", iterations, "val loss:", avg_val_loss, "time:", val_time_end - val_time_start)
                        patience += 1

        print()
        counter += 1

    rounds += 1
