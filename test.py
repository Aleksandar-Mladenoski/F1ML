import pandas as pd
import fastf1

df = pd.read_csv(r'G:\F1ML\F1Data\f1_data_combined.csv')

# To get the count of NA values in each column
na_per_column = df.isna().sum()
print(na_per_column[na_per_column > 0])  # Only displays columns with NA values

# To get the total number of NA values in the entire dataset
total_na = df.isna().sum().sum()
print(f"Total NA values in dataset: {total_na}")

# Drop the 'PitOutTime', 'PitInTime', and 'DeletedReason' columns
your_dataframe = df.drop(columns=['PitOutTime', 'PitInTime', 'DeletedReason'])

# Fill NA values in 'TrackStatus' with 0
your_dataframe['TrackStatus'].fillna(0, inplace=True)

# Drop the single row where 'LapTime' is missing
your_dataframe = your_dataframe.dropna(subset=['LapTime'])

# Check the result for remaining NA values
print(your_dataframe.isna().sum())  # To confirm remaining NA values

# Investigate rows where 'SpeedFL' is missing
missing_speedfl = your_dataframe[your_dataframe['SpeedFL'].isna()]
print(missing_speedfl)

# Drop the 'SpeedFL' column as well
your_dataframe = your_dataframe.drop(columns=['SpeedFL'])

# Check if there are any remaining NA values
print(your_dataframe.isna().sum())  # Confirm remaining NA values
your_dataframe.to_csv(r'G:\F1ML\F1Data\f1_data_clean.csv', index=False)

print(df.iloc[0])
