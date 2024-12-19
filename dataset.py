import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class FSDataset(Dataset):
    def __init__(self, lap_data_dir, final_results_dir):
        self.lap_data = pd.read_csv(lap_data_dir)
        self.final_results = pd.read_csv(final_results_dir)
        self.samples_per_year = dict()
        self.years = np.sort(self.lap_data['Year'].unique())
        # print(np.sum(pd.isna(self.lap_data).values))
        for year in self.years:
            self.samples_per_year.update({year: self.lap_data[self.lap_data['Year'] == year]['GP'].nunique()})

    def __len__(self):
        samples = 0
        for year in self.years:
            samples += self.lap_data[self.lap_data['Year'] == year]['GP'].nunique()
        return samples

    def __getitem__(self, idx):
        cum_sample = self.samples_per_year[self.years[0]]
        idx_within_year = 1
        for i, year in enumerate(self.years):
            if year == self.years[0]:
                idx_within_year = np.min(self.lap_data[self.lap_data['Year'] == year]['GP_num'].values)
            else:
                idx_within_year = 1
            if idx < cum_sample:
                idx_year = year
                idx_within_year += idx - (cum_sample - self.samples_per_year[self.years[i]])
                break
            cum_sample += self.samples_per_year[self.years[i+1]]


        laps = self.lap_data[(self.lap_data['Year'] == idx_year) & (self.lap_data['GP_num'] == idx_within_year)].copy()
        laps['PositionPerSession'] = (laps.groupby('Session')['LapStartTime'].rank(method='first')).apply(pd.to_numeric, errors='coerce')
        laps['PositionPerDriver'] = (laps.groupby('Driver')['LapStartTime'].rank(method='first')).apply(pd.to_numeric, errors='coerce')

        result = self.final_results[(self.final_results['Year'] == idx_year) & (self.final_results['GP_num'] == idx_within_year)]

        if idx_year == 2024 and idx_within_year == 3:
            result = pd.concat([result, pd.DataFrame([["SAR", 20.0, 2024, 'Australian Grand Prix', 3]], columns=result.columns)], ignore_index=True)
        laps.drop(['Year', 'GP', 'Session', 'Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'LapStartTime', 'LapStartDate', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime'], axis=1, inplace=True)
        #print(f"Year: {idx_year}, idx_within_year: {idx_within_year}")
        #print("Expected GP_num values:", self.final_results[self.final_results['Year'] == idx_year]['GP_num'].unique())

        if year == 2021 and idx_within_year == 22:
            # print('mame ti ebam mazespinska picka ti materina gomnarska')
            result = result.fillna(20, inplace=False)
            #print(result.to_string())
        if year == 2022 and idx_within_year == 2:
            # print('mercedes goldenboy more like mf fucking up my fucking training loop')
            result = result.fillna(20, inplace=False)
            #print(result.to_string())
        if year == 2023 and idx_within_year == 15:
            #print('WINDOWLICKER ODI EBISE MAME TI EBAM')
            result = result.fillna(20, inplace=False)
            #print(result.to_string())

        return laps, result, idx_year, idx_within_year

def collate_fn(batch, max_driver_laps_per_session: int = 219):
    driver_laps_batch = list()
    driver_to_idx_batch = list()
    result_batch = list()
    idx_years = list()
    idx_within_years = list()
    for i in range(len(batch)):
        laps, result, idx_year, idx_within_year = batch[i]
        drivers_laps_list = list()
        results = list()
        driver_to_idx_dict = dict()

        for idx, driver in enumerate(pd.unique(result['Abbreviation'].values)):
            driver_laps = laps[laps['Driver'] == driver].drop(['Driver'], axis=1).copy()
            temp = [([0] * driver_laps.shape[1]) for i in range(max_driver_laps_per_session-driver_laps.shape[0])]
            pad = pd.DataFrame(temp, columns=driver_laps.columns)
            driver_laps_padded = pd.concat([driver_laps, pad], axis=0, ignore_index=True)

            #print([f"{i}: {x}"for i, x in enumerate(driver_laps_padded.dtypes.values)])
            #print([f"{driver_laps.columns.values[i]}: {x}"for i, x in enumerate(driver_laps.iloc[0].values) ])

            driver_laps_tensor = torch.tensor(driver_laps_padded.to_numpy(), dtype=torch.float32)
            # print(result)
            # print(driver)
            driver_result_tensor = torch.tensor(result[result['Abbreviation'] == driver]['Position'].iloc[0].astype(float))


            driver_to_idx_dict[driver] = idx
            drivers_laps_list.append(driver_laps_tensor)
            results.append(driver_result_tensor)

        drivers_laps_tensor = torch.stack(drivers_laps_list)  # Shape: [num_drivers, max_driver_laps, feature_dim]
        drivers_results_tensor = torch.stack(results)  # Shape: [num_drivers]

        driver_laps_batch.append(drivers_laps_tensor)  # Append the stacked tensor
        result_batch.append(drivers_results_tensor)
        driver_to_idx_batch.append(driver_to_idx_dict)
        idx_years.append(idx_year)
        idx_within_years.append(idx_within_year)

    # After the loop, stack the batches
    driver_laps_batch = torch.stack(driver_laps_batch)  # Shape: [batch_size, num_drivers, max_driver_laps, feature_dim]
    result_batch = torch.stack(result_batch)  # Shape: [batch_size, num_drivers]



    return driver_laps_batch, result_batch, driver_to_idx_batch, idx_years, idx_within_years
        # temp = [([0] * laps.shape[1]) for i in range(3693-laps.shape[0])]
        # pad = pd.DataFrame(temp, columns=laps.columns)
        # laps_padded = pd.concat([laps, pad], axis=0, ignore_index=True)








def f1data_train_test_split(lap_data_dir, final_results_dir, test_precent: float, absolute_path: str = r'F1Data'):
    if (test_precent > 1) or (0 > test_precent):
        raise ValueError("Test_precent must be between 0 and 1")

    lap_data = pd.read_csv(lap_data_dir)
    final_results = pd.read_csv(final_results_dir)

    samples_per_year = dict()
    years = np.sort(lap_data['Year'].unique())

    num_gp = 0

    for year in years:
        num_gp_per_year = lap_data[lap_data['Year'] == year]['GP'].nunique()
        num_gp += num_gp_per_year
        samples_per_year.update({year: num_gp_per_year})

    print(num_gp)
    num_test_gp = int(test_precent * num_gp)
    num_train_gp = num_gp - num_test_gp

    cumulative_gp_num = 0
    for year in years:
        cumulative_gp_num += samples_per_year[year]
        if num_train_gp < cumulative_gp_num:
            stopping_year = year
            stopping_gp_within_year = num_train_gp - (cumulative_gp_num-samples_per_year[year])
            gp_num_within_year = num_train_gp - (cumulative_gp_num - samples_per_year[year])

            break

    training_lap_data = lap_data[(lap_data['Year'] < stopping_year) | ( (lap_data['Year'] == stopping_year) & (lap_data['GP_num'] <= gp_num_within_year))]
    training_results_data = final_results[(final_results['Year'] < stopping_year) | ( (final_results['Year'] == stopping_year) & (final_results['GP_num'] <= gp_num_within_year))]

    test_lap_data = lap_data[(lap_data['Year'] > stopping_year) | ( (lap_data['Year'] == stopping_year) & (lap_data['GP_num'] > gp_num_within_year))]
    test_results_data = final_results[(final_results['Year'] > stopping_year) | ( (final_results['Year'] == stopping_year) & (final_results['GP_num'] > gp_num_within_year))]


    training_lap_data.to_csv(absolute_path +r'\training_lap_data.csv', index=False)
    training_results_data.to_csv(absolute_path +r'\training_results_data.csv', index=False)

    test_lap_data.to_csv(absolute_path+ r'\test_lap_data.csv', index=False)
    test_results_data.to_csv(absolute_path+ r'\test_results_data.csv', index=False)







#f1data_train_test_split(r'F1Data\f1_preprocessed.csv', r'F1Data\f1_final_result_data.csv', 0.1)


# train_dataset = FSDataset(r'F1Data\training_lap_data.csv', r'F1Data\training_results_data.csv')
# test_dataset = FSDataset(r'F1Data\test_lap_data.csv', r'F1Data\test_results_data.csv')
#
# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=4, collate_fn=collate_fn)
# for batch in train_loader:
#     driver_laps_batch, result_batch, driver_to_idx_batch, _, _ = batch
#     print(driver_laps_batch)

# for sample in train_dataset:
#     print(sample[0].shape)
#
# for sample in test_dataset:
#     print(sample[0].shape)
#
# print(f"Length of training data: {len(train_dataset)},Length of testing data:  {len(test_dataset)}")


# print(len(dataset)) # 144


# print(dataset[0][0].iloc[0])
# print(len(dataset[0][0].columns.values))
# print(dataset[0][1])


# max_laps_per_driver = 0
# for gp in dataset:
#     laps, results, idx_year, idx_within_year = gp
#     for driver in pd.unique(laps['Driver'].values):
#         driver_laps = laps[laps['Driver'] == driver]
#         max_laps_per_driver = len(driver_laps) if len(driver_laps) > max_laps_per_driver else max_laps_per_driver
#
# print(max_laps_per_driver) # = 219
