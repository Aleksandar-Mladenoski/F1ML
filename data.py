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


def impute_columns(df, column):
    avg_column = df.groupby('Driver')[column].mean().to_dict()

    df[column+'_IsImputed'] = df.apply(
        lambda row: pd.isna(row[column]), axis=1
    )

    df[column] = df.apply(
        lambda row: avg_column[row['Driver']] if pd.isna(row[column]) else row[column], axis=1
    )

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


def main():


    year = 2024
    gp = 11
    event = fastf1.get_event(year, gp)
    sessions = list(potential_sessions.intersection(event.values)) # How we can get the potential sessions in an event
    session_name = 'Race'
    session = event.get_session(session_name)
    session.load()
    print(sessions[0])

    # Code to get LapNorm for each driver in each session
    lap_data = session.laps
    drivers = set(session.laps['Driver'].values)
    max_lap_per_driver_per_session = lap_data.groupby('Driver')['LapNumber'].max().to_dict()
    avg_sec1_per_driver = lap_data.groupby('Driver')['Sector1Time'].mean().to_dict()
    print(max_lap_per_driver_per_session)

    lap_data['NaN_count'] = lap_data.isna().sum(axis=1)

    lap_data = lap_data[lap_data['NaN_count'] <= 2].copy()

    lap_data.drop(columns=['NaN_count'], inplace=True)

    lap_data['LapNorm'] = lap_data['LapNumber']/(lap_data['Driver'].map(max_lap_per_driver_per_session))
    lap_data["Session"] = session_name
    lap_data["Position"] = 0 if lap_data['Position'].isna().all() else lap_data['Position']


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


    lap_data['PitLap'] = lap_data.apply(
        lambda row: False if pd.isna(row['PitOutTime']) and pd.isna(row["PitInTime"]) else True, axis=1
    )
    lap_data.drop(['DriverNumber', 'DeletedReason', 'PitOutTime', 'PitInTime'], axis=1, inplace=True)
    lap_data['Year'] = year
    lap_data['GP'] = gp
    print(lap_data.isna().any())
    print(lap_data.iloc[0])
    print(lap_data.columns.values)
if __name__ == "__main__":
    main()