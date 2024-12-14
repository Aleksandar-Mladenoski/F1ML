# Session:
    # 1. drivers() - List of all the drivers, returns list
        # get_driver() - Get driver, returns DriverResult, subclass of DataSeries
    # 2. results - Get the results, returns SessionResults, subclass of DataFrame

    # Main dataset which I will focus on, maybe in the future, I can use telemetry data to predict lap times and such, for now though, this is the main deal
    # 3. laps - All laps of all drivers within this session, returns Laps subclass of DataFrame

    # Temporal data based on measurements of gap 1 minute, could be used to crossreference laps and append air temp and humidity etc etc
    # 4. weather_data() - Get the weather data, returns DataFrame directly

    # Temporal data of RPMs, Speed and other car technical things. No way to incorporate into the laps, for now I will leave it
    # 5. car_data() - Dictionary of car telemetry data, by string of car number, ex: "1", returns Dict

    # Temporal data for position, X Y Z
    # 6. pos_data() - Positionary data of the car, for each car, returns Dictionary

    # 7. .session_status - start, finish timestamps, returns DataFrame

    # 8. .session_start_time - Session start time. returns datetime.timedelta

    # New idea, we train on all the laps and we include LapNumber / Max Laps in order to indicate where this lap is located in the sting, so we don't loose the development of the race, it makes sense why
    # The start lap might be slower than at the end, less fuel and potentially better tyre compound etc etc

# ['Time' 'Driver' 'DriverNumber' 'LapTime' 'LapNumber' 'Stint' 'PitOutTime'
#  'PitInTime' 'Sector1Time' 'Sector2Time' 'Sector3Time'
#  'Sector1SessionTime' 'Sector2SessionTime' 'Sector3SessionTime' 'SpeedI1'
#  'SpeedI2' 'SpeedFL' 'SpeedST' 'IsPersonalBest' 'Compound' 'TyreLife'
#  'FreshTyre' 'Team' 'LapStartTime' 'LapStartDate' 'TrackStatus' 'Position'
#  'Deleted' 'DeletedReason' 'FastF1Generated' 'IsAccurate' 'LapNorm'
#  'Session']

# Time - fine
# Driver - fine
# DriverNumber - kind of useless, I would probably remove this. Perhaps just do dummies/one-hot encoding for Driver
# LapTime - I believe is useful
# Stint - Still useful
# Pit out Pit in I think is probably overkill, perhaps just make a combination feature called "IsPitLap" which says True or False whether it is a pit lap or not
#

import pandas as pd
import fastf1

potential_sessions = {"Practice 1", "Practice 2", "Practice 3", "Sprint Qualifying", "Qualifying", "Sprint Race", "Race"}
session_per_year = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 20}

fastf1.Cache.enable_cache('G:\F1ML\F1Data')
fastf1.Cache.offline_mode('enabled')

def impute_columns(df, column):
    df[column].fillna(df[column].mean(), inplace=True)  # Example using mean


def impute_columns_sessiontime(df, column):
    df[column + '_IsImputed'] = df.apply(
        lambda row: pd.isna(row[column]), axis=1
    )

    if column == 'Sector1SessionTime':
        avg_s1 = df.groupby('Driver')['Sector1Time'].mean().to_dict()
        df[column] = df.apply(
            lambda row: row["Sector2SessionTime"] - avg_s1[row['Driver']], axis=1
        )
    elif column == 'Sector2SessionTime':
        avg_s2 = df.groupby('Driver')['Sector2Time'].mean().to_dict()
        df[column] = df.apply(
            lambda row: row["Sector1SessionTime"] + avg_s2[row['Driver']], axis=1
        )
    else:
        avg_s3 = df.groupby('Driver')['Sector3Time'].mean().to_dict()
        df[column] = df.apply(
            lambda row: row["Sector2SessionTime"] + avg_s3[row['Driver']], axis=1
        )


def add_weather_data(df):
    air_temp = list()
    humidity = list()
    pressure = list()
    rainfall = list()
    track_temp = list()
    wind_direction = list()
    wind_speed = list()


    for idx, lap in df.iterrows():

        weather_data = lap.get_weather_data()

        air_temp.append(weather_data['AirTemp'])
        humidity.append(weather_data['Humidity'])
        pressure.append(weather_data['Pressure'])
        rainfall.append(weather_data['Rainfall'])
        track_temp.append(weather_data['TrackTemp'])
        wind_direction.append(weather_data['WindDirection'])
        wind_speed.append(weather_data['WindSpeed'])


    df['AirTemp'] = air_temp
    df['Humidity'] = humidity
    df['Pressure'] = pressure
    df['Rainfall'] = rainfall
    df['TrackTemp'] = track_temp
    df['WindDirection'] = wind_direction
    df['WindSpeed'] = wind_speed


def main():
    year_dfs = list()
    result_dfs = list()
    for year in range(2018,2024+1): # 2018,2024+1
        weekend_dfs = list()
        for weekend in range(1, session_per_year[year]+1): # 1, session_per_year[year]+1
            event = fastf1.get_event(year, weekend)
            sessions = list(potential_sessions.intersection(event.values)) # How we can get the potential sessions in an event

            session_dfs = list()
            for session_name in sessions:
                if year == 2019 and weekend == 17 and session_name == 'Practice 3': # Actually cancelled
                    continue
                elif year == 2020 and weekend == 2 and session_name == 'Practice 3': # Actually cancelled
                    continue
                elif year == 2020 and weekend == 11 and (session_name == 'Practice 1' or session_name == 'Practice 2'): # Actually cancelled
                    continue
                elif year == 2021 and weekend == 1 and session_name == 'Practice 1': #  Not cancelled
                    continue
                elif year == 2021 and weekend == 15 and session_name == 'Practice 3': # Actually cancelled
                    continue
                elif year == 2022 and weekend == 14 and session_name == 'Qualifying':  # Not cancelled
                    continue
                elif year == 2022 and weekend == 6 and session_name == 'Practice 2': # Not cancelled, but 3.4.1 still doesn't work.
                    continue
                elif year == 2022 and weekend == 11 and session_name == 'Practice 1': # Not cancelled, but 3.4.1 still doesn't work.
                    continue
                elif year == 2022 and weekend == 16 and session_name == 'Qualifying': # Not cancelled, but 3.4.1 still doesn't work
                    continue
                elif year == 2022 and weekend == 21 and session_name == 'Qualifying': # Not cancelled, but 3.4.1 still doesn't work
                    continue
                elif year == 2023 and weekend == 23 and session_name == 'Practice 2': # Not cancelled but, 3.4.1 still doesn't work
                    continue
                elif year == 2024 and weekend == 8 and session_name == 'Qualifying': #Not cancelled
                    continue
                elif year == 2024 and weekend == 17 and session_name == 'Qualifying': #Not cancelled
                    continue


                # elif year == 2023 and weekend == 6: # Actually cancelled
                #     continue

                if year == 2024 and weekend == 20:
                    fastf1.Cache.offline_mode('disabled')

                session = event.get_session(session_name)
                session.load()
                weekend_name = session.event['EventName']

                if session_name == 'Race':
                    result = session.results
                    result = result.drop(['DriverNumber', 'BroadcastName', 'DriverId', 'TeamName',
                                          'TeamColor', 'TeamId', 'FirstName', 'LastName', 'FullName',
                                          'HeadshotUrl', 'CountryCode', 'ClassifiedPosition',
                                          'GridPosition', 'Q1', 'Q2', 'Q3', 'Time', 'Status', 'Points'], axis=1)
                    result['Year'] = year
                    result['GP'] = weekend_name
                    result['GP_num'] = weekend
                    result_dfs.append(result)

                # print(weekend_name)
                # print(sessions[0])



                # Code to get LapNorm for each driver in each session
                try:
                    lap_data = session.laps
                except fastf1.core.DataNotLoadedError:
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    print(year, weekend, session_name)
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    continue
                drivers = set(session.laps['Driver'].values)
                max_lap_per_driver_per_session = lap_data.groupby('Driver')['LapNumber'].max().to_dict()
                # print(max_lap_per_driver_per_session)

                lap_data['LapNorm'] = lap_data['LapNumber']/(lap_data['Driver'].map(max_lap_per_driver_per_session))
                lap_data["Session"] = session_name
                lap_data['Position'].fillna(0, inplace=True)
                lap_data['TrackStatus'].fillna(0, inplace=True)


                lap_data['PitLap'] = lap_data.apply(
                    lambda row: False if pd.isna(row['PitOutTime']) and pd.isna(row["PitInTime"]) else True, axis=1
                )
                lap_data['Year'] = year
                lap_data['GP'] = weekend_name
                lap_data['Session'] = session_name
                lap_data['GP_num'] = weekend

                try:
                    add_weather_data(lap_data)
                except fastf1.core.DataNotLoadedError:
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    print(year, weekend, session_name)
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    print("HELLLLLLLLOOOOOOOOOOOOOOOOOO")
                    continue
                lap_data.drop(['PitOutTime', 'PitInTime', 'DeletedReason'], axis=1, inplace=True)

                lap_data['NaN_count'] = lap_data.isna().sum(axis=1)

                lap_data = lap_data[lap_data['NaN_count'] < 2   ].copy()

                lap_data.drop(columns=['NaN_count'], inplace=True)

                impute_columns(lap_data, 'Sector1Time')
                impute_columns(lap_data, 'Sector2Time')
                impute_columns(lap_data, 'Sector3Time')
                impute_columns(lap_data, 'SpeedI1')
                impute_columns(lap_data, 'SpeedI2')
                impute_columns(lap_data, 'SpeedFL')
                impute_columns(lap_data, 'SpeedST')
                impute_columns(lap_data, 'Sector1SessionTime')
                impute_columns(lap_data, 'Sector2SessionTime')
                impute_columns(lap_data, 'Sector3SessionTime')

                #lap_data.drop(['DriverNumber', 'DeletedReason'], axis=1, inplace=True)

                # print(lap_data.isna().any())
                # print(lap_data.iloc[0])
                # print(lap_data.columns.values)
                session_dfs.append(lap_data)

            if not (len(session_dfs) == 0):
                weekend_dfs.append(pd.concat(session_dfs, axis=0, ignore_index=True))
        year_dfs.append(pd.concat(weekend_dfs, axis=0, ignore_index=True))

    final_result_data = pd.concat(result_dfs, axis=0, ignore_index=True)
    final_data = pd.concat(year_dfs, axis=0, ignore_index=True)
    final_result_data.to_csv(r'G:\F1ML\F1Data\f1_final_result_data.csv', index=False)
    final_data.to_csv(r'G:\F1ML\F1Data\f1_data_combined.csv', index=False)



if __name__ == "__main__":
    main()



# For future notice, I should incorporate telemetry data in the predictions, not sure how right now, but yeah
# Same as get_weather_data(), method for each lap that spits out your telemetry data, get_telemetry() for non cached, telemtry for cached. Optional Freq option
# 'Date', 'SessionTime', 'DriverAhead', 'DistanceToDriverAhead', 'Time',
#        'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source',
#        'Distance', 'RelativeDistance', 'Status', 'X', 'Y', 'Z'],

# DO NOT FORGET, WHEN ACCESSING TELEMTRY DATA OR CAR POS DATA OR WHATEVER, THE METHOD USES DRIVER NUMBER TO SEARCH FOR THE LAP. DO NOT DROP THIS BEFORE YOU ARE COMPLETELY FINISHED WITH ORGANISING THE DATASET.

# Column values above.