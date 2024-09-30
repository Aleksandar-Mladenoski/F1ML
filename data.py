import pandas as pd
import fastf1

potential_sessions = {"Practice 1", "Practice 2", "Practice 3", "Sprint Qualifying", "Qualifying", "Sprint Race", "Race"}

def main():
    event = fastf1.get_event(2024, 11)
    sessions = list(potential_sessions.intersection(event.values)) # How we can get the potential sessions in an event
    fp1 = event.get_session("Practice 1")
    fp1.load()
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

    # Idea is, I guess we're doing this on a  5 grand prix weekend basis, all 5 events for the past 4, for the last fifth we do only 4 events ( except the race of course )
    # Not a per lap basis, rather the most important metrics for all the laps for all drivers.
    # Average, Standard Deviation, Variance, 25th Percentile, Median, 75th Percentile
    # Each of these lap entries should be per lap ( 3 entries per session ) grand prix weekend ( for the differential in track ) per session ( usually qualifying laps are fastest ) per driver ( each driver has their own unique capabilities ) per red bull
    #pos_data = pd.DataFrame(s1.pos_data["1"])
    #print(pos_data.columns)
    #print(pos_data.iloc[0].values)
    #print(pos_data.iloc[1].values)

    # car_data = pd.DataFrame(s1.car_data["1"])
    # print(car_data.columns)
    # print(car_data)


    # Time is measured per session, starts at 0, use this for nearest addition for the laps (aka 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
    #        'WindDirection', 'WindSpeed' )
    # weather_data = fp1.weather_data
    # print(weather_data.columns)
    # print(weather_data)

    # Index(['Time', 'Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint',
    #        'PitOutTime', 'PitInTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
    #        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
    #        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
    #        'Compound', 'TyreLife', 'FreshTyre', 'Team', 'LapStartTime',
    #        'LapStartDate', 'TrackStatus', 'Position', 'Deleted', 'DeletedReason',
    #        'FastF1Generated', 'IsAccurate'],

    # for each laptime we take, Time, Stint, TyreLife, Compound, Sector1Time, Sector2Time, Sector3Time,
    lap_data = fp1.laps
    print(lap_data.columns)
    print(lap_data)

    interested_laps = list()
    interested_laps.append(lap_data[lap_data['Driver'] == "VER"]['LapTime'].min())
    interested_laps.append(lap_data[lap_data['Driver'] == "VER"]['LapTime'].quantile(.25))
    interested_laps.append(lap_data[lap_data['Driver'] == "VER"]['LapTime'].quantile(.50))
    interested_laps.append(lap_data[lap_data['Driver'] == "VER"]['LapTime'].quantile(.75))
    interested_laps.append(lap_data[lap_data['Driver'] == "VER"]['LapTime'].max())

    print(lap_data[lap_data['Driver'] == "VER"]['LapTime'].quantile([.25, .50, .75]))

    print(interested_laps)

    laps_ver = [lap_data[lap_data['LapTime'] == (lap_data[lap_data['Driver'] == "VER"]['LapTime'].min())]]
    print(laps_ver)

    print(lap_data['TyreLife'].unique())

if __name__ == "__main__":
    main()