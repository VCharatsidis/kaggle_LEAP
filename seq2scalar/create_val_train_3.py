import polars as pl

train_file = '../data/train.csv'

val_1 = pl.read_csv('../data/validation_set.csv')
val_2 = pl.read_csv('../data/validation_set_2.csv')
val_3 = pl.read_csv('../data/validation_set_3.csv')
val_5 = pl.read_csv('../data/validation_set_5.csv')
val_6 = pl.read_csv('../data/validation_set_6.csv')
val_7 = pl.read_csv('../data/validation_set_7.csv')

sample_ids = (list(val_1['sample_id'].unique()) +
              list(val_2['sample_id'].unique()) +
              list(val_3['sample_id'].unique()) +
              list(val_5['sample_id'].unique()) +
              list(val_6['sample_id'].unique()) +
                list(val_7['sample_id'].unique()))

chunk_size = 4000
reader = pl.read_csv_batched(train_file, batch_size=chunk_size)

new_train_file = '../data/train_set_8.csv'
new_val_file = '../data/validation_set_8.csv'

first_chunk = True
counter = 0
while True:

    try:
        batches = reader.next_batches(100)
        if batches is None:
            break  # No more data to read
    except:
        break

    for df in batches:
        filtered_df = df.filter(~pl.col("sample_id").is_in(sample_ids))

        sampled_df = filtered_df.sample(n=int(0.2*chunk_size))
        val_sample_ids = list(sampled_df['sample_id'].unique())

        val_data = df.filter(pl.col("sample_id").is_in(val_sample_ids))
        train_data = df.filter(~pl.col("sample_id").is_in(val_sample_ids))

        print(counter, "train_data:", train_data.shape, "val_data:", val_data.shape)

        # Write train chunk with header management
        mode = 'w' if first_chunk else 'a'
        train_data.to_pandas().to_csv(new_train_file, mode=mode, header=first_chunk, index=False)
        val_data.to_pandas().to_csv(new_val_file, mode=mode, header=first_chunk, index=False)
        first_chunk = False
        counter += 1


