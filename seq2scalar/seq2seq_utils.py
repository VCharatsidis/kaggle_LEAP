import time

import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import TARGET_WEIGHTS, ERR, min_std
from neural_net.utils import r2_score


def to_seq2seq_tensor(vertically_resolved_variables, scalar_variables):
    # Read the CSV file
    chunk_size = 2000000

    reader = pl.read_csv_batched('../data/train.csv', batch_size=chunk_size)
    batches = reader.next_batches(5)

    df = batches[0]

    # Initialize an empty list to hold the tensors
    batch_size = len(df)
    sequence_length = 60
    num_variables = len(vertically_resolved_variables) + len(scalar_variables)
    data = np.zeros((batch_size, sequence_length, num_variables))

    # Process vertically resolved variables
    for i, var in enumerate(vertically_resolved_variables):
        columns = [f"{var}_{level}" for level in range(sequence_length)]
        data[:, :, i] = df[columns].to_numpy()

    # Process scalar variables
    for j, var in enumerate(scalar_variables):
        data[:, :, len(vertically_resolved_variables) + j] = df[var].to_numpy()[:, np.newaxis]

    # Convert the numpy array to a PyTorch tensor
    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Print the shape to verify
    print(tensor_data.shape)  # Should print torch.Size([batch_size, 60, 25])
    return tensor_data


def collate_fn(batch):
    data, targets = zip(*batch)
    data = torch.stack(data, dim=0)
    targets = torch.stack(targets, dim=0)
    return data, targets


def to_tensor(df, batch_size, sequence_length, seq_variables_x, scalar_variables_x):
    num_variables = len(seq_variables_x) + len(scalar_variables_x)

    data_X = np.zeros((batch_size, sequence_length, num_variables))

    # Process vertically resolved variables
    for i, var in enumerate(seq_variables_x):
        columns = [f"{var}_{level}" for level in range(sequence_length)]
        data_X[:, :, i] = df[columns].to_numpy()

    # Process scalar variables
    for j, var in enumerate(scalar_variables_x):
        data_X[:, :, len(seq_variables_x) + j] = df[var].to_numpy()[:, np.newaxis]

    # Convert the numpy array to a PyTorch tensor
    tensor_data = torch.tensor(data_X, dtype=torch.float32).cuda()

    return tensor_data


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_model(model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, mean_y, std_y):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_time_start = time.time()
        for src, tgt in val_loader:
            val_preds = model(src)
            val_preds = val_preds.cpu()

            tgt = tgt.cpu().numpy()
            tgt = ((tgt * std_y) + mean_y) * TARGET_WEIGHTS

            val_preds[:, std_y < (1.1 * min_std)] = 0
            val_preds = val_preds.numpy()
            val_preds = ((val_preds * std_y) + mean_y) * TARGET_WEIGHTS

            tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
            val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

            val_loss += r2_score(val_preds, tgt).item()

        val_time_end = time.time()

        avg_val_loss = val_loss / len(val_loader)
        #scheduler.step()  # Adjust learning rate

        if avg_val_loss < min_loss:
            print("epoch:", epoch, "model saved", "chunk:", counter, "iterations:", iterations, "val loss:",
                  avg_val_loss, "time:", val_time_end - val_time_start)
            torch.save(model, f"models/{model_name}")
            min_loss = avg_val_loss
            patience = 0
        else:
            print("epoch:", epoch, f"No improvement in validation loss for {patience} epochs.", "chunk:", counter,
                  "iterations:", iterations, "val loss:", avg_val_loss, "time:", val_time_end - val_time_start)
            patience += 1

    return patience, min_loss


def seq2scalar_32(weighted, df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y):

    # Preprocess the features and target columns
    for col in FEAT_COLS:
        X = df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float64))
    for col in TARGET_COLS:
        y = df.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float64))

    # Normalize features
    X = X.with_columns([(pl.col(col) - mean_x[i]) / std_x[i] for i, col in enumerate(FEAT_COLS)])

    # Reshape the features into the desired shape [batch_size, 60, 25]
    batch_size = X.shape[0]
    sequence_length = 60

    # Convert the numpy array to a PyTorch tensor
    tensor_data = to_tensor(X, batch_size, sequence_length, seq_variables_x, scalar_variables_x)
    print("tensor_data input shape:", tensor_data.shape)
    y = y.to_numpy()
    if weighted:
        y = y * TARGET_WEIGHTS

    y = (y - mean_y) / std_y
    tensor_target = torch.tensor(y, dtype=torch.float32).cuda()

    # Create an instance of the dataset
    dataset = CustomDataset(tensor_data, tensor_target)

    return dataset, tensor_data
