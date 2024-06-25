import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from torch.utils.data import DataLoader
from decimal import Decimal

from glu_mlp import GLU_MLP
from neural_net.utils import mlp_data_32, count_parameters, get_val_loss, collate_fn
from constants import TARGET_WEIGHTS
from transformers import get_linear_schedule_with_warmup


def preprocess():
    train_file = '../data/train_set_2.csv'
    # Read only the first row (header) to get column names
    df_header = pl.read_csv(train_file, has_header=True, skip_rows=0, n_rows=100)
    FEAT_COLS = df_header.columns[1:557]
    TARGET_COLS = df_header.columns[557:]

    mean_x = np.load("../data/means_x.npy")
    std_x = np.load("../data/std_x.npy")
    mean_y = np.load("../data/means_y.npy")
    std_y = np.load("../data/std_y.npy")

    min_std = 1e-12
    std_y = np.clip(std_y, a_min=min_std, a_max=None)
    std_x = np.clip(std_x, a_min=min_std, a_max=None)

    print(mean_x.shape, std_x.shape)
    print(mean_y.shape, std_y.shape)

    validation_size = 1091032
    print(validation_size)

    # validation_size = pl.read_csv().shape[0]

    # Number of columns and their names
    column_names = df_header.columns
    num_columns = len(column_names)
    print("num columns:", num_columns)

    chunk_size = 100000  # Define the size of each batch

    start_from = 0
    if start_from > 0:
        print("WARNING: Starting from chunk:", start_from)

    input_dim = len(FEAT_COLS)
    hidden_dim = 1000
    num_layers = 9
    output_dim = len(TARGET_COLS)
    LEARNING_RATE = 0.7e-4
    BATCH_SIZE = 256

    #model = GLU_MLP(input_dim, hidden_dim, output_dim, num_layers).cuda()

    model_name = f'glu_mlp_{min_std}_l_{num_layers}_d_{hidden_dim}_weightless_train_set_2'

    model = torch.load(f"models/{model_name}.model")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_warmup_steps = 100
    num_training_steps = 800000

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return model, optimizer, scheduler, mean_x, std_x, mean_y, std_y, FEAT_COLS, TARGET_COLS, validation_size, chunk_size, model_name, LEARNING_RATE, BATCH_SIZE, min_std, TARGET_WEIGHTS, train_file


def memory_eff_eval(validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE, model):
    val_file = "../data/validation_set_2.csv"
    val_reader = pl.read_csv_batched(val_file, batch_size=chunk_size//2)

    all_preds = np.zeros((validation_size, len(TARGET_WEIGHTS)))
    all_targets = np.zeros((validation_size, len(TARGET_WEIGHTS)))
    start_idx = 0
    sum_rows = 0
    while True:
        try:
            val_df = val_reader.next_batches(1)[0]
            sum_rows += val_df.shape[0]

            if val_df is None:
                break  # No more data to read
        except:
            break

        val_dataset, _ = mlp_data_32(val_df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y)

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                )

        all_preds, all_targets, start_idx = get_val_loss(model, all_preds, all_targets, start_idx, mean_y, std_y, val_loader)

    all_preds *= TARGET_WEIGHTS
    all_targets *= TARGET_WEIGHTS

    #mean_all_set = np.load("../data/means_y_after_norm.npy")
    mean_all_targets = np.mean(all_targets, axis=0)

    # for i, col in enumerate(TARGET_COLS):
    #     print(col, mean_all_set[i], mean_all_targets[i])
    # input()

    ss_res_vector = (all_targets - all_preds) ** 2
    ss_tot_vector = (all_targets - mean_all_targets) ** 2

    ss_res = np.sum(ss_res_vector)
    ss_tot = np.sum(ss_tot_vector)

    # min_preds = all_preds.min(axis=0)
    # max_preds = all_preds.max(axis=0)
    #
    # min_targets = all_targets.min(axis=0)
    # max_targets = all_targets.max(axis=0)
    # mean_targets = np.mean(all_targets, axis=0)

    denom_r2 = np.sum(ss_tot_vector, axis=0) * TARGET_WEIGHTS + (1 - TARGET_WEIGHTS)
    #mean_denom_r2 = np.mean(ss_tot_vector, axis=0) * TARGET_WEIGHTS + (1 - TARGET_WEIGHTS)
    mean_r2 = np.sum(ss_res_vector, axis=0) / denom_r2

    for i in range(len(TARGET_COLS)):
        print(f"Target {i} {TARGET_COLS[i]} R2:", mean_r2[i], TARGET_WEIGHTS[i])

    print("MSE:", np.mean(ss_res_vector), "mean R2:", np.mean(mean_r2), "Fake R2:", ss_res/ss_tot)

    loss = np.mean(mean_r2)

    return loss


def eval_cross_attention_weightless(model, min_loss, patience, epoch, counter, iterations, model_name, validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE):
    model.eval()

    val_time_start = time.time()
    loss = memory_eff_eval(validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE, model)
    val_time_end = time.time()

    if loss < min_loss:
        print("epoch:", epoch, "model saved", "chunk:", counter, "iterations:", iterations, "val loss:",
              loss, "time:", val_time_end - val_time_start)
        torch.save(model, f"models/{model_name}.model")
        min_loss = loss
        patience = 0
    else:
        print("epoch:", epoch, f"No improvement in validation loss for {patience} chunks.", "chunk:", counter,
              "iterations:", iterations, "val loss:", loss, "time:", val_time_end - val_time_start)
        patience += 1

    return patience, min_loss, loss


def train():
    model, optimizer, scheduler, mean_x, std_x, mean_y, std_y, FEAT_COLS, TARGET_COLS, validation_size, chunk_size, model_name, LEARNING_RATE, BATCH_SIZE, min_std, TARGET_WEIGHTS, train_file = preprocess()

    start_eval = time.time()
    min_loss = memory_eff_eval(validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE, model)  # memory_eff_eval()
    end_eval = time.time()
    print("Initial validation loss:", min_loss, "time:", end_eval - start_eval)
    time.sleep(1)

    # tensor_mean_y = torch.tensor(mean_y).cuda()
    # tensor_std_y = torch.tensor(std_y).cuda()

    patience = 0
    r2_denom = np.load("../data/r2_denominator.npy")
    r2_denom = r2_denom * TARGET_WEIGHTS + (1 - TARGET_WEIGHTS)
    # for i, col in enumerate(TARGET_COLS):
    #     print(col, r2_denom[i], TARGET_WEIGHTS[i])

    r2_denom = torch.tensor(r2_denom).cuda()

    Targets_norm = torch.tensor(TARGET_WEIGHTS).cuda()
    # criterion = nn.MSELoss()  # Using MSE for regression
    epoch = 0
    iterations = 0
    while True:
        reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
        counter = 0

        prep_chunk_time_start = time.time()
        try:
            batches = reader.next_batches(20)
            if batches[0] is None:
                break  # No more data to read
        except:
            break

        end_batches = time.time()
        print("end batches time:", end_batches - prep_chunk_time_start)

        for df in batches:

            start_chunk_time = time.time()
            train_dataset, _ = mlp_data_32(df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y)

            train_loader = DataLoader(train_dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      collate_fn=collate_fn,
                                      )

            prep_chunk_time_end = time.time()

            print("epoch:", epoch, "chunk:", counter, "prep chunk time:", prep_chunk_time_end - start_chunk_time)

            total_loss = 0
            steps = 0
            train_time_start = time.time()

            model.train()

            for batch_idx, (src, tgt) in enumerate(train_loader):

                optimizer.zero_grad()
                preds = model(src)

                # preds = preds * tensor_std_y + tensor_mean_y
                # tgt = tgt * tensor_std_y + tensor_mean_y

                tgt = tgt * Targets_norm
                preds = preds * Targets_norm

                ss_res = (tgt - preds) ** 2
                mean_ss_res = torch.mean(ss_res, dim=0)

                r2 = mean_ss_res / r2_denom

                loss = r2.mean()
                # loss = torch.sum(ss_res) / torch.sum(ss_tot)
                #loss = criterion(preds, tgt)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()  # Update learning rate

                total_loss += loss.item()

                iterations += 1
                steps += 1

                if (batch_idx + 1) % 100 == 0:
                    train_time_end = time.time()
                    print("epoch:", epoch, f', chunk: {counter}, Step {batch_idx + 1}, Training Loss: {total_loss / steps:.4f}', "iterations:", iterations, "time:", train_time_end - train_time_start)
                    total_loss = 0  # Reset the loss for the next steps
                    steps = 0  # Reset step count
                    #time.sleep(4)

            counter += 1

        patience, min_loss, _ = eval_cross_attention_weightless(model, min_loss, patience, epoch, counter, iterations, model_name, validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE)

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, end of chunk {counter}, Learning Rate: {param_group['lr']}")
        print()

    epoch += 1


train()
