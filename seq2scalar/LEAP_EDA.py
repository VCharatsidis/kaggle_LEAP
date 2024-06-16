import polars as pl
import matplotlib.pyplot as plt

from constants import TARGET_WEIGHTS

# Load data from a CSV file
df = pl.read_csv('../data/train.csv')

FEAT_COLS = df.columns[1:557]
TARGET_COLS = df.columns[557:]

# df = df.with_columns(
#     [pl.col(column) * weight for column, weight in zip(TARGET_COLS, TARGET_WEIGHTS)]
# )


skewness = df.select([pl.col(column).skew().alias(f"skewness_{column}") for column in FEAT_COLS])
for column in FEAT_COLS:
    sk_col = f"skewness_{column}"
    print(f"Skewness of {column}: {skewness[sk_col].to_numpy()[0]}")

# Histogram and box plot for each column
for column in FEAT_COLS:
    data = df[column].to_pandas()  # Convert to pandas for plotting

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data.dropna(), bins=30)  # Drop NA for plotting
    plt.title(f'Histogram of {column}')
    histogram_path = f"plots/histogram_train_{column}.png"  # Define path to save histogram
    plt.savefig(histogram_path)  # Save the histogram plot
    plt.close()  # Close the plot to avoid display issues

    # plt.subplot(1, 2, 2)
    # plt.boxplot(data.dropna())  # Drop NA for plotting
    # plt.title(f'Box Plot of {column}')
    #
    # plt.show()