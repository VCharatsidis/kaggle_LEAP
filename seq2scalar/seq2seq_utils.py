import time

import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import TARGET_WEIGHTS
from neural_net.utils import r2_score
from transformer_constants import scalar_vars_num, vector_vars_num


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


def to_tensor_flat(df, batch_size, features):

    data = np.zeros((batch_size, len(features), 1))

    for idx, col in enumerate(features):
        data[:, idx, 0] = df[col].to_numpy()

    # Convert the numpy array to a PyTorch tensor
    tensor_data = torch.tensor(data, dtype=torch.float32).cuda()

    return tensor_data


def to_tensor(df, batch_size, sequence_length, seq_variables, scalar_variables):
    num_variables = len(seq_variables) + len(scalar_variables)

    data = np.zeros((batch_size, sequence_length, num_variables))

    # Process vertically resolved variables
    for i, var in enumerate(seq_variables):
        columns = [f"{var}_{level}" for level in range(sequence_length)]
        data[:, :, i] = df[columns].to_numpy()

    # Process scalar variables
    for j, var in enumerate(scalar_variables):
        data[:, :, len(seq_variables) + j] = df[var].to_numpy()[:, np.newaxis]

    # Convert the numpy array to a PyTorch tensor
    tensor_data = torch.tensor(data, dtype=torch.float32).cuda()

    return tensor_data


def to_tensor_rev(df, batch_size, sequence_length, seq_variables_x, scalar_variables_x):
    num_variables = len(seq_variables_x) + len(scalar_variables_x)

    # Initialize the numpy array with shape (batch_size, num_variables, sequence_length)
    data_X = np.zeros((batch_size, num_variables, sequence_length))

    # Process vertically resolved variables
    for i, var in enumerate(seq_variables_x):
        columns = [f"{var}_{level}" for level in range(sequence_length)]
        data_X[:, i, :] = df[columns].to_numpy()

    # Process scalar variables
    for j, var in enumerate(scalar_variables_x):
        data_X[:, len(seq_variables_x) + j, :] = df[var].to_numpy()[:, np.newaxis]

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


def mean_and_flatten(preds):
    vector_vars = preds[:, :, :vector_vars_num]
    scalar_vars = preds[:, :, -scalar_vars_num:]

    # Compute the mean along dim=1, excluding the first 6 that are vector vars
    mean_vars = scalar_vars.mean(dim=1, keepdim=True).squeeze(dim=1)
    vector_vars = vector_vars.reshape(vector_vars.size(0), -1)  # shape: [batch_size, seq_len * d_model]

    concat = torch.cat((vector_vars, mean_vars), dim=1)

    return concat


def eval_model_seq_2_seq(min_std, weighted, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, mean_y, std_y):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_time_start = time.time()
        for src, tgt in val_loader:
            val_preds = model(src, tgt)
            val_preds = val_preds.cpu()

            val_preds = mean_and_flatten(val_preds)
            tgt = mean_and_flatten(tgt)

            tgt = tgt.cpu().numpy()
            if not weighted:
                tgt = ((tgt * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                tgt = (tgt * std_y) + mean_y

            val_preds[:, std_y < (1.1 * min_std)] = 0
            val_preds = val_preds.numpy()

            if not weighted:
                val_preds = ((val_preds * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                val_preds = (val_preds * std_y) + mean_y

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


def eval_generate_model_seq_2_seq(up_to, min_std, weighted, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, mean_y, std_y):
    model.eval()
    with torch.no_grad():
        val_loss = []
        val_time_start = time.time()
        for idx, (src, tgt) in enumerate(val_loader):
            if idx % 50 == 0:
                print("val idx:", idx)

            if idx > up_to:
                break

            val_preds = model.generate(src, 60)
            val_preds = val_preds.cpu()

            val_preds = mean_and_flatten(val_preds)
            tgt = mean_and_flatten(tgt)

            tgt = tgt.cpu().numpy()
            if not weighted:
                tgt = ((tgt * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                tgt = (tgt * std_y) + mean_y

            val_preds[:, std_y < (1.1 * min_std)] = 0
            val_preds = val_preds.numpy()

            if not weighted:
                val_preds = ((val_preds * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                val_preds = (val_preds * std_y) + mean_y

            tgt = torch.tensor(tgt, dtype=torch.float64).cuda()
            val_preds = torch.tensor(val_preds, dtype=torch.float64).cuda()

            this_val_loss = r2_score(val_preds, tgt).item()
            val_loss.append(this_val_loss)

        val_time_end = time.time()

        avg_val_loss = sum(val_loss) / len(val_loss)
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

    model.train()
    return patience, min_loss


def eval_model(min_std, weighted, model, val_loader, min_loss, patience, epoch, counter, iterations, model_name, mean_y, std_y):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_time_start = time.time()
        for src, tgt in val_loader:
            val_preds = model(src)
            val_preds = val_preds.cpu()

            tgt = tgt.cpu().numpy()
            if not weighted:
                tgt = ((tgt * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                tgt = (tgt * std_y) + mean_y

            val_preds[:, std_y < (1.1 * min_std)] = 0
            val_preds = val_preds.numpy()

            if not weighted:
                val_preds = ((val_preds * std_y) + mean_y) * TARGET_WEIGHTS
            else:
                val_preds = (val_preds * std_y) + mean_y

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


def seq2seq_32(min_std, df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y):

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

    y = y.with_columns([(pl.col(col) * TARGET_WEIGHTS[i] - mean_y[i]) / std_y[i] for i, col in enumerate(TARGET_COLS)])

    # y = y.with_columns([
    #         pl.when(std_y[i] < (1.1 * min_std))
    #         .then(0)
    #         .otherwise(((pl.col(col) * TARGET_WEIGHTS[i] - mean_y[i]) / std_y[i]) )
    #         .alias(col)
    #         for i, col in enumerate(TARGET_COLS)
    # ])

    tensor_target = to_tensor(y, batch_size, sequence_length, seq_variables_y, scalar_variables_y)

    # Create an instance of the dataset
    dataset = CustomDataset(tensor_data, tensor_target)

    return dataset, tensor_data


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


def seq2scalar_flat(weighted, df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y):

    # Preprocess the features and target columns
    for col in FEAT_COLS:
        X = df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float64))
    for col in TARGET_COLS:
        y = df.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float64))

    # Normalize features
    X = X.with_columns([(pl.col(col) - mean_x[i]) / std_x[i] for i, col in enumerate(FEAT_COLS)])

    # Reshape the features into the desired shape [batch_size, 60, 25]
    batch_size = X.shape[0]

    # Convert the numpy array to a PyTorch tensor
    tensor_data = to_tensor_flat(df, batch_size, FEAT_COLS)
    print("tensor_data input shape:", tensor_data.shape)
    y = y.to_numpy()
    if weighted:
        y = y * TARGET_WEIGHTS

    y = (y - mean_y) / std_y
    tensor_target = torch.tensor(y, dtype=torch.float32).cuda()

    # Create an instance of the dataset
    dataset = CustomDataset(tensor_data, tensor_target)

    return dataset, tensor_data


def seq2scalar_32_rev(weighted, df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y):

    # Preprocess the features and target columns
    for col in FEAT_COLS:
        X = df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float64))
    for col in TARGET_COLS:
        y = df.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float64))

    # Normalize features
    X = X.with_columns([(pl.col(col) - mean_x[i]) / std_x[i] for i, col in enumerate(FEAT_COLS)])

    # Reshape the features into the desired shape [batch_size, 60, 25]
    batch_size = X.shape[0]
    atmospheric_levels = 60

    # Convert the numpy array to a PyTorch tensor
    tensor_data = to_tensor_rev(X, batch_size, atmospheric_levels, seq_variables_x, scalar_variables_x)
    print("tensor_data input shape:", tensor_data.shape)
    y = y.to_numpy()
    if weighted:
        y = y * TARGET_WEIGHTS

    y = (y - mean_y) / std_y
    tensor_target = torch.tensor(y, dtype=torch.float32).cuda()

    # Create an instance of the dataset
    dataset = CustomDataset(tensor_data, tensor_target)

    return dataset, tensor_data
