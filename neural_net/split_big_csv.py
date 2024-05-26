import polars as pl

from constants import total_rows

file_path = '../data/train.csv'

chunk_size = 1000000
train_path = '../data/train_set_2.csv'
val_path = '../data/validation_set_2.csv'

print("total rows:", total_rows)
train_fraction = 0.9
train_rows = int(train_fraction * total_rows)
processed_rows = 0

# Initialize writing, manage headers explicitly
first_chunk = True
counter = 0

reader = pl.read_csv_batched(file_path, batch_size=chunk_size)
batches = reader.next_batches(10)

num_train_per_chunk = int(train_fraction * chunk_size)

for df in batches:
    print("df shape:", df.shape)

    if processed_rows < train_rows:

        val_size = chunk_size - num_train_per_chunk
        train_chunk = df.slice(val_size, chunk_size)
        val_chunk = df.slice(0, val_size)

        print("train_chunk:", train_chunk.shape, "val_chunk:", val_chunk.shape)

        # Write train chunk with header management
        mode = 'w' if first_chunk else 'a'
        train_chunk.to_pandas().to_csv(train_path, mode=mode, header=first_chunk, index=False)

        # Write validation chunk with header management
        if not val_chunk.is_empty():
            val_chunk.to_pandas().to_csv(val_path, mode=mode, header=first_chunk, index=False)
    else:
        # Write remaining rows to validation
        df.to_pandas().to_csv(val_path, mode='a', header=first_chunk, index=False)

    processed_rows += len(df)
    first_chunk = False  # Turn off header writing after the first chunk
    print("circle done!", counter)
    counter += 1
