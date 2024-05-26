import polars as pl

# Example DataFrame creation
df = pl.DataFrame({
    "A": range(1, 101),
    "B": range(101, 201)
})

# Define the size of the first slice
total_rows = df.height
first_part_size = int(0.7 * total_rows)

# Slice the first part
first_part = df.slice(0, first_part_size)

# Slice the second part
second_part = df.slice(first_part_size, total_rows - first_part_size)

# Verify the parts
print("First Part:")
print(first_part)
print("\nSecond Part:")
print(second_part)
