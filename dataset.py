import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FSDataset(Dataset):
    def __init__(self, lap_data_dir, final_results_dir):
        self.lap_data = pd.read_csv(lap_data_dir, index_col=0)
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
            if idx < cum_sample:
                idx_year = year
                idx_within_year += idx - (cum_sample - self.samples_per_year[self.years[i]])
                break
            cum_sample += self.samples_per_year[self.years[i+1]]

        laps = self.lap_data[(self.lap_data['Year'] == idx_year) & (self.lap_data['GP_num'] == idx_within_year)].copy()
        laps['PositionPerSession'] = laps.groupby('Session')['LapStartTime'].rank(method='first').astype(int)
        laps['PositionPerDriver'] = laps.groupby('Driver')['LapStartTime'].rank(method='first').astype(int)
        result = self.final_results[(self.final_results['Year'] == idx_year) & (self.final_results['GP_num'] == idx_within_year)]

        if idx_year == 2024 and idx_within_year == 3:
            result = pd.concat([result, pd.DataFrame([["SAR", 20.0, 2024, 'Australian Grand Prix', 3]], columns=result.columns)], ignore_index=True)
        laps.drop(['Year', 'GP', 'Session', 'Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time'], axis=1, inplace=True)
        #print(f"Year: {idx_year}, idx_within_year: {idx_within_year}")
        #print("Expected GP_num values:", self.final_results[self.final_results['Year'] == idx_year]['GP_num'].unique())

        return laps, result, idx_year, idx_within_year

def collate_fn(batch, max_driver_laps_per_session: int = 219):
    driver_laps_batch = list()
    result_batch = list()
    idx_years = list()
    idx_within_years = list()
    for i in range(len(batch)):
        laps, result, idx_year, idx_within_year = batch[i]
        driver_laps_dict = dict()
        for driver in pd.unique(laps['Driver'].values):
            driver_laps = laps[laps['Driver'] == driver]
            temp = [([0] * driver_laps.shape[1]) for i in range(max_driver_laps_per_session-laps.shape[0])]
            pad = pd.DataFrame(temp, columns=driver_laps.columns)
            driver_laps_padded = pd.concat([driver_laps, pad], axis=0, ignore_index=True)
            driver_laps_tensor = torch.tensor(driver_laps_padded.to_numpy(), dtype=torch.float32)
            driver_laps_dict.update({driver: driver_laps_tensor})
        result.drop(['Year', 'GP', 'GP_num'], axis=1, inplace=True) # Columns before: 'Abbreviation' 'Position' 'Year' 'GP' 'GP_num'
        result_batch.append(result)
        idx_years.append(idx_year)
        idx_within_years.append(idx_within_year)
        driver_laps_batch.append(driver_laps_dict)

    return driver_laps_batch, result_batch, idx_years, idx_within_years
        # temp = [([0] * laps.shape[1]) for i in range(3693-laps.shape[0])]
        # pad = pd.DataFrame(temp, columns=laps.columns)
        # laps_padded = pd.concat([laps, pad], axis=0, ignore_index=True)



dataset = FSDataset(r'F1Data\f1_preprocessed.csv', r'F1Data\f1_final_result_data.csv')
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



dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
for batch in dataloader:
    driver_laps_batch, result_batch, idx_years, idx_within_years = batch
    first_laps_data = driver_laps_batch[0]
    print(result_batch[0].columns.values)
    # print(first_laps_data.keys()) # ['GAS', 'ALO', 'LEC', 'STR', 'VAN', 'MAG', 'HUL', 'HAR', 'RIC', 'VER', 'SIR', 'HAM', 'VET', 'SAI', 'RAI', 'BOT', 'GRO', 'ERI', 'PER', 'OCO']
    ver_laps = first_laps_data['VER']
    print(pd.unique(ver_laps['LapTime_converted'].values))
    break