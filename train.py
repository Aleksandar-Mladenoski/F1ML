import pandas as pd
import torch, torch.nn as nn
import transformer
#from dataset import FSDataset


lap_data = pd.read_csv(r'F1Data\f1_preprocessed.csv')
result = pd.read_csv(r'F1Data\f1_final_result_data.csv')

print(result.columns.values)

# dataset = FSDataset(r'F1Data\f1_dropped_na.csv', r'F1Data\f1_final_result_data.csv')
# epoch_num = 5
# batch_size = 4
# learning_rate = 0.001
#
# for epoch in range(epoch_num):
