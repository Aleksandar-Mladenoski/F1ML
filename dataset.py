import pandas as pd
import matplotlib
import torch
import torch.utils.data.dataset as Dataset
import numpy as np

# df_clean = pd.read_csv(r'F1Data\f1_dropped_na.csv')
# print(df_clean.head)
# print(f"NA Values in df: {df_clean.isnan().values.sum()}")

lap_data = pd.read_csv(r'F1Data\f1_dropped_na.csv')
print(lap_data['GP'].count())
years = np.sort(lap_data['Year'].unique())
print(years)


class FSDataset(Dataset):
    def __init__(self, lap_data_dir, final_results_dir):
        self.lap_data = pd.read_csv(lap_data_dir)
        self.final_results_dir = pd.read_csv(final_results_dir)

    def __len__(self):
        samples = 0
        self.samples_per_year = dict()
        self.years = np.sort(self.lap_data['Year'].unique())
        for year in years:
            self.samples_per_year.update({year: self.lap_data[lap_data['Year'] == year]['GP'].nunique()})
            samples += self.lap_data[lap_data['Year'] == year]['GP'].nunique()
        return samples

    def __getitem__(self, idx):
        cum_sample = self.samples_per_year[self.years[0]]
        idx_year = 0
        for i, year in enumerate(self.years):
            if idx <= cum_sample:
                idx_year = year
                break
            cum_sample += self.samples_per_year[self.years[i+1]]
