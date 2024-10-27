import pandas as pd
import fastf1

df = pd.read_csv(r'G:\F1ML\F1Data\f1_data_combined.csv')
# To get the count of NA values in each column
na_per_column = df.isna().sum()
print(na_per_column[na_per_column > 0])  # Only displays columns with NA values

# To get the total number of NA values in the entire dataset
#total_na = df.isna().sum().sum()
#print(f"Total NA values in dataset: {total_na}")
#print(df.where(pd.isna(df[''])))

