import fastf1
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv(r'F1Data\f1_dropped_na.csv')
event_2024 =  fastf1.get_session(2024, 20, 'R')
event_2024.load()
drivers = ['ALB', 'ALO', 'BOT', 'COL', 'GAS', 'HAM', 'HUL', 'LAW', 'MAG', 'NOR', 'OCO', 'PER', 'PIA', 'RUS', 'SAI', 'STR', 'TSU', 'VER', 'ZHO']


# Filter for years 2019 to 2024
df_filtered = df[(df['Year'] >= 2019) & (df['Year'] <= 2024)]
df_filtered = df_filtered[df_filtered['Driver'].isin(drivers)]
print('Drivers filtered!')
print(df_filtered)
# Group by GP, Year, Session, and Driver to get the lap counts for each session
lap_counts = df_filtered.groupby(['GP', 'Year', 'Session', 'Driver']).size().reset_index(name='LapCount')

# Sum the lap counts for all sessions within each event (GP, Year) for each driver
event_lap_counts = lap_counts.groupby(['GP', 'Year', 'Driver'])['LapCount'].sum().reset_index()

# Find the minimum lap count for each event (across all drivers) to ensure consistency
min_lap_counts_per_event = event_lap_counts.groupby(['GP', 'Year'])['LapCount'].min().reset_index()

print("Minimum lap count required for each event:")
print(min_lap_counts_per_event.min())
print(type(min_lap_counts_per_event))
plt.hist(min_lap_counts_per_event['LapCount'], bins=20)
plt.show()




















# # Group by GP, Year, Session, and Driver to get the lap counts for each session
# lap_counts = df_filtered.groupby(['GP', 'Year', 'Session', 'Driver']).size().reset_index(name='LapCount')
#
# # Sum the lap counts for all sessions within each event (GP, Year) for each driver
# event_lap_counts = lap_counts.groupby(['GP', 'Year', 'Driver'])['LapCount'].sum().reset_index()
#
# # Find the minimum lap count for each event (across all drivers) to ensure consistency
# min_lap_counts_per_event = event_lap_counts.groupby(['GP', 'Year'])['LapCount'].min().reset_index()
#
# print("Minimum lap count required for each event:")
# print(min_lap_counts_per_event)
# event_lap_counts.groupby(['GP', 'Year'])['LapCount'].hist(bins=30)
# plt.show()

# event = fastf1.get_event(2024, 21)
#
# session_fp1 = event.get_session('Practice 1')
# session_fp1.load()
# flap = session_fp1.laps
# flap.get_weather_data()
#
# session_fp2 = event.get_session('Practice 2')
# session_fp2.load()
# flap = session_fp2.laps
# flap.get_weather_data()
#
# session_fp3 = event.get_session('Practice 3')
# session_fp3.load()
# flap = session_fp3.laps
# flap.get_weather_data()
#
# session_q = event.get_session('Qualifying')
# session_q.load()
# flap = session_q.laps
# flap.get_weather_data()
#
# session_r = event.get_session('Race')
# session_r.load()
# flap = session_r.laps
# flap.get_weather_data()
