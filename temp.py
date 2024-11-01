import fastf1
import pandas as pd

# To get the count of NA values in each column
your_dataframe = pd.read_csv(r'G:\F1ML\F1Data\f1_data_clean.csv')
print('Length of dataframe: ', len(your_dataframe))
print('How many laps I have from Qualifying 2024', len(your_dataframe.where(your_dataframe['Year'] == 2024 )))

na_per_column = your_dataframe.isna().sum()
print(na_per_column[na_per_column > 0])  # Only displays columns with NA values

# To get the total number of NA values in the entire dataset
total_na = your_dataframe.isna().sum().sum()
print(f"Total NA values in dataset: {total_na}")
