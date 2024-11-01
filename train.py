import fastf1
import pandas as pd
import sklearn


df = pd.read_csv(r'G:\F1ML\F1Data\f1_data_combined.csv')
max_interlagos_2023 = df[(df['GP'] == 'São Paulo Grand Prix')]
print(max_interlagos_2023)
print(df['GP'].value_counts())
print(df.isna().sum())






















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
