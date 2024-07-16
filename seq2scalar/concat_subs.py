import polars as pl

df1 = pl.read_csv('submissions/cross_fixed_set_1.csv')
df2 = pl.read_csv('submissions/cross_weightless_set_2.csv')
df3 = pl.read_csv('submissions/cross_fixed_set_3.csv')
df5 = pl.read_csv('submissions/cross_fixed_set_5.csv')
df6 = pl.read_csv('submissions/cross_fixed_set_6.csv')
df7 = pl.read_csv('submissions/cross_fixed_set_7.csv')

cols_to_avg = [col for col in df1.columns if col != 'sample_id']

# Calculate the average of the selected columns
average_df = df1.select(cols_to_avg) + df2.select(cols_to_avg) + df3.select(cols_to_avg) + df5.select(cols_to_avg) + df6.select(cols_to_avg) + df7.select(cols_to_avg)
average_df = average_df / 6

# Add the excluded column back to the averaged DataFrame
average_df = pl.concat([df1.select('sample_id'), average_df], how="horizontal")

average_df.write_csv('submissions/concat_sub_1_2_3_5_6_7_new.csv')
