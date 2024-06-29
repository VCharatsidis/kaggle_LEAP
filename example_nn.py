import polars as pl

# Example CSV content:
# sample_id, value
# 1, 100
# 2, 200
# 3, 300

# Read the CSV file
df = pl.read_csv('data/validation_set.csv')

# Extract all values from the 'sample_id' column
sample_ids = df['sample_id'].to_list()

print(len(sample_ids))

import json
# Write the list to a JSON file
with open('sample_ids_val.json', 'w') as f:
    json.dump(sample_ids, f)
