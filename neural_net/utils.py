import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    data, targets = zip(*batch)
    data = torch.stack(data, dim=0)
    targets = torch.stack(targets, dim=0)
    return data, targets


def get_val_loss(model, all_preds, all_targets, start_idx, means, stds, val_loader):
    with torch.no_grad():
        for src, tgt in val_loader:
            val_preds = model(src)

            val_preds = val_preds.cpu().numpy()
            tgt = tgt.cpu().numpy()

            val_preds = val_preds * stds + means
            tgt = tgt * stds + means

            batch_size = tgt.shape[0]
            all_targets[start_idx: start_idx + batch_size] = tgt
            all_preds[start_idx: start_idx + batch_size] = val_preds
            start_idx += batch_size

        return all_preds, all_targets, start_idx



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def mlp_data_32(df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y):
    # Preprocess the features and target columns
    for col in FEAT_COLS:
        X = df.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))
    for col in TARGET_COLS:
        y = df.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float32))

    X, y = X.to_numpy(), y.to_numpy()

    # Normalize features
    X = (X - mean_x) / std_x
    y = (y - mean_y) / std_y

    tensor_data = torch.tensor(X, dtype=torch.float32).cuda()
    tensor_target = torch.tensor(y, dtype=torch.float32).cuda()

    # Create an instance of the dataset
    dataset = CustomDataset(tensor_data, tensor_target)

    return dataset, tensor_data


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        """
        Initialize with NumPy arrays.
        """
        assert x.shape[0] == y.shape[0], "Features and labels must have the same number of samples"
        self.x = x
        self.y = y

    def __len__(self):
        """
        Total number of samples.
        """
        return self.x.shape[0]

    def __getitem__(self, index):
        """
        Generate one sample of data.
        """
        # Convert the data to tensors when requested
        return torch.from_numpy(self.x[index]).float().cuda(), torch.from_numpy(self.y[index]).float().cuda()


def calc_x_mean_and_std(file, FEAT_COLS, min_std):
    data = pl.read_csv(file, columns=FEAT_COLS)

    for col in FEAT_COLS:
        X = data.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))

    X = X.to_numpy()
    mx = X.mean(axis=0)
    sx = np.maximum(X.std(axis=0), min_std)

    np.save('mean_x_2.npy', mx)
    np.save('std_x_2.npy', sx)
    print("done calculating mean and std for X")


def calc_y_mean_and_std(file, TARGET_COLS, min_std):
    data = pl.read_csv(file, columns=TARGET_COLS)

    for col in TARGET_COLS:
        y = data.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float32))

    y = y.to_numpy()

    my = y.mean(axis=0)
    sy = np.maximum(np.sqrt((y * y).mean(axis=0)), min_std)

    np.save('../data/mean_y.npy', my)
    np.save('../data/std_y.npy', sy)
    print("done")


def mae(y_true, y_pred):
    """
    Calculate mean absolute error
    """
    return torch.mean(torch.abs(y_true - y_pred))


def r2_score(y_pred, y_true):
    """
    Calculate R-squared score
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = ss_res / ss_tot
    return r2


def get_val_loader(file, FEAT_COLS, TARGET_COLS, mean_y, std_y, mean_x, std_x, batch_size):
    val_data = pl.read_csv(file)
    print("val_data shape:", val_data.shape)
    for col in FEAT_COLS:
        val_X = val_data.select(FEAT_COLS).with_columns(pl.col(col).cast(pl.Float32))
    for col in TARGET_COLS:
        val_y = val_data.select(TARGET_COLS).with_columns(pl.col(col).cast(pl.Float32))

    val_X, val_y = val_X.to_numpy(), val_y.to_numpy()
    val_y = (val_y - mean_y.reshape(1, -1)) / std_y.reshape(1, -1)

    # norm X
    val_X = (val_X - mean_x.reshape(1, -1)) / std_x.reshape(1, -1)

    val_dataset = NumpyDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return val_loader
