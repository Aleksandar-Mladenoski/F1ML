import fastf1
import pandas as pd

# To get the count of NA values in each column
df = pd.read_csv(r'F1Data\f1_dropped_na.csv')

session = fastf1.get_session(2024, 20, 'R')
session.load()

result = session.results
result = result.drop(['DriverNumber', 'BroadcastName', 'DriverId', 'TeamName',
       'TeamColor', 'TeamId', 'FirstName', 'LastName', 'FullName',
       'HeadshotUrl', 'CountryCode', 'ClassifiedPosition',
       'GridPosition', 'Q1', 'Q2', 'Q3', 'Time', 'Status', 'Points'], axis=1)
print(result)