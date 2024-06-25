import polars as pl

# Example DataFrame
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
})

# Get the data type of the 'age' column
print(df["age"].dtype)

print(df.schema)