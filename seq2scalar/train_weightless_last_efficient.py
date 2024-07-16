
import time

import numpy as np
import torch
import torch.optim as optim
import polars as pl
from torch.utils.data import DataLoader

from cross_attention_fix import CrossAttentionModel_2
from mlp_assemmble import CustomModel
from complex_model import Complex_Transformer
from hoyso_model import Hoyso_Transformer
from modified_seq_to_scalar_positional import ModifiedSequenceToScalarTransformer_positional
from neural_net.utils import r2_score
from seq2seq_utils import seq2scalar_32, count_parameters, collate_fn, get_val_loss_cross_attention_weightless
from constants import seq_variables_x, scalar_variables_x, seq_variables_y, scalar_variables_y, seq_length, \
    input_variable_order, TARGET_WEIGHTS

from transformers import get_linear_schedule_with_warmup

set = 8

def preprocess():
    train_file = f'../data/train_set_{set}.csv'
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

    val_file = f"../data/validation_set_{set}.csv"
    validation_size = pl.scan_csv(val_file).collect().shape[0]
    print("validation size:", validation_size)

    # validation_size = pl.read_csv().shape[0]

    # Number of columns and their names
    column_names = df_header.columns
    num_columns = len(column_names)
    print("num columns:", num_columns)

    chunk_size = 100000  # Define the size of each batch

    start_from = 0
    if start_from > 0:
        print("WARNING: Starting from chunk:", start_from)

    feature_dim = 25
    d_model = 256
    nhead = 8
    num_encoder_layers = 6
    dim_feedforward = 256
    output_dim = 368
    dropout = 0.0
    LEARNING_RATE = 1.9e-4
    BATCH_SIZE = 256

    model_name = f'cross_attention_fixed_1e-12_nhead_8_enc_l_{num_encoder_layers}_d_256_weightless_train_set_{set}'
    #model = CrossAttentionModel_2(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()

    #model = Complex_Transformer(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()
    #model = Hoyso_Transformer(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()
    #model_name = f'Hoyso_transfromer_{min_std}_nhead_{nhead}_enc_l_{num_encoder_layers}_d_{d_model}_weightless_train_set_{set}'
    #model_name = f"mlp_assemble_output_dim_50_final_output_dim=800_num_layers_8"

    model = torch.load(f"models/{model_name}.model")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    #optimizer.load_state_dict(torch.load(f'optimizer_state_set_{set}.pth'))

    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE
        param_group['initial_lr'] = LEARNING_RATE

    for param_group in optimizer.param_groups:
        print(f"Learning Rate: {param_group['lr']}")

    num_training_steps = 210000

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-9)

    # Assuming 'optimizer' is your optimizer object

    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Param group {i}:")
        for key in param_group.keys():
            if key != 'params':
                print(f"  {key}: {param_group[key]}")

    return model, optimizer, scheduler, mean_x, std_x, mean_y, std_y, FEAT_COLS, TARGET_COLS, validation_size, chunk_size, model_name, LEARNING_RATE, BATCH_SIZE, min_std, TARGET_WEIGHTS, train_file


def memory_eff_eval(validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE, model, verbose=False):
    val_file = f"../data/validation_set_{set}.csv"
    val_reader = pl.read_csv_batched(val_file, batch_size=chunk_size)

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

        val_dataset, _ = seq2scalar_32(val_df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x, scalar_variables_x)

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                )

        all_preds, all_targets, start_idx = get_val_loss_cross_attention_weightless(model, all_preds, all_targets, start_idx, mean_y, std_y, val_loader)

    all_preds *= TARGET_WEIGHTS
    all_targets *= TARGET_WEIGHTS

    mean_all_targets = np.mean(all_targets, axis=0)

    ss_res_vector = (all_targets - all_preds) ** 2
    ss_tot_vector = (all_targets - mean_all_targets) ** 2

    ss_res = np.sum(ss_res_vector)
    ss_tot = np.sum(ss_tot_vector)

    denom_r2 = np.sum(ss_tot_vector, axis=0) * TARGET_WEIGHTS + (1 - TARGET_WEIGHTS)
    mean_r2 = np.sum(ss_res_vector, axis=0) / denom_r2

    if verbose:
        for i in range(len(TARGET_COLS)):
            print(f"Target {i} {TARGET_COLS[i]} R2:", mean_r2[i], TARGET_WEIGHTS[i])

    print("MSE:", np.mean(ss_res_vector), "mean R2:", np.mean(mean_r2), "Fake R2:", ss_res/ss_tot)

    loss = np.sum(mean_r2) / np.sum(TARGET_WEIGHTS)

    return loss


def eval_cross_attention_weightless(optimizer, model, min_loss, patience, epoch, counter, iterations, model_name, validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE):
    model.eval()

    val_time_start = time.time()
    loss = memory_eff_eval(validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE, model)
    val_time_end = time.time()

    if loss < min_loss:
        print("epoch:", epoch, "model saved", "chunk:", counter, "iterations:", iterations, "val loss:",
              loss, "time:", val_time_end - val_time_start)
        torch.save(model, f"models/{model_name}.model")
        torch.save(optimizer.state_dict(), f'optimizer_state_set_{set}.pth')
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
    min_loss = memory_eff_eval(validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE, model, True)  # memory_eff_eval()
    end_eval = time.time()
    print("Initial validation loss:", min_loss, "time:", end_eval - start_eval)

    tensor_std_y = torch.tensor(std_y).cuda()
    tensor_mean_y = torch.tensor(mean_y).cuda()

    patience = 0
    #r2_denom = np.load("../data/r2_denom_denorm.npy")
    r2_denom = np.load("../data/r2_denominator.npy")
    r2_denom = r2_denom * TARGET_WEIGHTS + (1 - TARGET_WEIGHTS)
    r2_denom = torch.tensor(r2_denom).cuda()

    Targets_norm = torch.tensor(TARGET_WEIGHTS).cuda()
    epoch = 0
    iterations = 0
    while True:

        reader = pl.read_csv_batched(train_file, batch_size=chunk_size)
        counter = 0

        while True:

            prep_batches_time_start = time.time()
            try:

                read_num = 30

                batches = reader.next_batches(read_num)
                if batches is None:
                    break  # No more data to read
            except:
                break

            print("prep batches time:", time.time() - prep_batches_time_start, "chunk:", counter)

            for df in batches:
                prep_chunk_time_start = time.time()
                train_dataset, _ = seq2scalar_32(df, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, seq_variables_x, scalar_variables_x)

                train_loader = DataLoader(train_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          collate_fn=collate_fn,
                                          )

                prep_chunk_time_end = time.time()

                print("epoch:", epoch, "chunk:", counter, "prep chunk time:", prep_chunk_time_end - prep_chunk_time_start)

                steps = 0
                train_time_start = time.time()

                model.train()
                total_loss = 0
                star_iteration = iterations

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
                    loss = torch.sum(r2) / torch.sum(Targets_norm)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    scheduler.step()  # Update learning rate

                    total_loss += loss.item()

                    iterations += 1
                    steps += 1

                    if (batch_idx + 1) % 100 == 0:
                        train_time_end = time.time()
                        print("epoch:", epoch, f', chunk: {counter}, Step {batch_idx + 1}, Training Loss: {total_loss / (iterations - star_iteration):.4f}', "iterations:", iterations, "time:", train_time_end - train_time_start)
                        steps = 0  # Reset step count

                counter += 1

            patience, min_loss, loss = eval_cross_attention_weightless(optimizer, model, min_loss, patience, epoch, counter, iterations, model_name, validation_size, chunk_size, FEAT_COLS, TARGET_COLS, mean_x, std_x, mean_y, std_y, BATCH_SIZE)
            for param_group in optimizer.param_groups:
                print(f"Epoch {epoch}, end of chunk {counter}, Learning Rate: {param_group['lr']}")
                print()

        epoch += 1


train()
