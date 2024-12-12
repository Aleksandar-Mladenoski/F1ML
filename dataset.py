import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class FSDataset(Dataset):
    def __init__(self, lap_data_dir, final_results_dir):
        self.lap_data = pd.read_csv(lap_data_dir)
        self.final_results = pd.read_csv(final_results_dir)
        self.samples_per_year = dict()
        self.years = np.sort(self.lap_data['Year'].unique())
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

        laps = self.lap_data[(self.lap_data['Year'] == idx_year) & (self.lap_data['GP_num'] == idx_within_year)]
        temp = [([0] * laps.shape[1]) for i in range(3693-laps.shape[0])]
        pad = pd.DataFrame(temp, columns=laps.columns)
        laps_padded = pd.concat([laps, pad], axis=0, ignore_index=True)
        result = self.final_results[(self.final_results['Year'] == idx_year) & (self.final_results['GP_num'] == idx_within_year)]

        if idx_year == 2024 and idx_within_year == 3:
            result = pd.concat([result, pd.DataFrame([["SAR", 20.0, 2024, 'Australian Grand Prix', 3]], columns=result.columns)], ignore_index=True)
        #print(f"Year: {idx_year}, idx_within_year: {idx_within_year}")
        #print("Expected GP_num values:", self.final_results[self.final_results['Year'] == idx_year]['GP_num'].unique())

        return laps_padded, result, idx_year, idx_within_year



dataset = FSDataset(r'F1Data\f1_dropped_na.csv', r'F1Data\f1_final_result_data.csv')
print(dataset[0][0])
print(len(dataset[0][0].columns.values))
print(dataset[0][1])