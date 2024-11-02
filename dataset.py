import pandas as pd
import matplotlib
import torch
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
        idx_year = 0
        for i, year in enumerate(self.years):
            if idx <= cum_sample:
                idx_year = year
                idx_within_year = idx - (cum_sample - self.samples_per_year[self.years[i]])
                break
            cum_sample += self.samples_per_year[self.years[i+1]]

        laps = self.lap_data[(self.lap_data['Year'] == idx_year) & (self.lap_data['GP_num'] == idx_within_year)]
        result = self.final_results[(self.final_results['Year'] == idx_year) & (self.final_results['GP_num'] == idx_within_year)]

        return laps, result



dataset = FSDataset(r'F1Data\f1_dropped_na.csv', r'F1Data\f1_final_result_data.csv')

for sample in dataset:
    print(sample[0])
    print(sample[1])