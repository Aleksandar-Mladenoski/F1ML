# import pandas as pd
# import numpy as np
#
# laps = pd.read_csv(r'F1Data\f1_dropped_na.csv',  index_col=0)
# result = pd.read_csv(r'F1Data\f1_final_result_data.csv')
# print(laps.columns.values)
#
#
# # Drop Unnamed, DriverNumber, Drop GP_num
# # Dummies for Compund, TrackStatus, Team, Year, GP, Session
# laps.drop(['DriverNumber'], axis=1, inplace=True)
# dummies = pd.get_dummies(laps, columns=['Compound', 'Team', 'Year', 'GP', 'Session'], drop_first=False)
# laps_padded = pd.concat([laps, dummies], axis=1)
#
#
# laps_padded.drop(['Compound', 'Team'], axis=1, inplace=True)
#
#
# print(laps_padded.columns.values)
# print(laps_padded.iloc[0])
#
# laps_padded.to_csv(r'F1Data\f1_preprocessed.csv')

import pandas as pd


def preprocess_datetime_features(df):
    # Columns with durations as time strings
    duration_cols = ['Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in duration_cols:
        # Convert string time to timedelta in seconds
        df[col+"_converted"] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()

    # Columns with start time or date
    datetime_cols = ['LapStartTime', 'LapStartDate']
    for col in datetime_cols:
        # Convert datetime to seconds since a reference point
        try:
            ref_point = pd.to_datetime(df[col], errors='coerce').min()  # Choose the earliest timestamp
            df[col+"_converted"] = (pd.to_datetime(df[col], errors='coerce') - ref_point).dt.total_seconds()
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            df[col+"_converted"] = None

    return df




laps = pd.read_csv(r'F1Data\f1_dropped_na.csv', index_col=0)
laps.drop(['DriverNumber'], axis=1, inplace=True)
categorical_cols = ['Compound', 'Team', 'Year', 'GP', 'Session']
dummies = pd.get_dummies(laps[categorical_cols], drop_first=False)
laps.drop(['Compound', 'Team'], axis=1, inplace=True)
laps_padded = pd.concat([laps, dummies], axis=1)
cols_to_keep = ['Year', 'GP', 'Session']

laps_padded = laps_padded.loc[:, ~laps_padded.columns.duplicated(keep='first')]

assert 'Year.1' not in laps_padded.columns, "Duplicate column 'Year.1' still exists!"

laps_padded = preprocess_datetime_features(laps_padded)

laps_padded.to_csv(r'F1Data\f1_preprocessed.csv')