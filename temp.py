import fastf1
import pandas as pd
session = fastf1.get_event(2024, 11)
practice1 = session.get_practice(1)
practice1.load()
weather_data = practice1.weather_data

print(weather_data.head())
