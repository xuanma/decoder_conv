"""
This class is designed to load M1 neural firing rates and EMG data for pytorch applications.
All spike and EMG data collected within one day are stored in a list.
Data corresponding to one file are stored in a numpy array.
"""
from torch.utils.data import Dataset
from util import flatten_list, flatten_list_3d
import torch
import numpy as np

def create_samples_xy_rnn(input_x, input_y, lags, transpose = 0):
    dataX, dataY = [], []
    for i in range(len(input_x) - lags):
        sample_X = input_x[i:i+lags, :]
        if transpose == 1:
            sample_X = sample_X.T
        sample_Y = input_y[i+lags-1, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    
    dataX = np.array(dataX)  # (N, lag, D)
    dataY = np.array(dataY)  # (N, D)
    return dataX, dataY

def create_samples_xy_rnn_list(input_x_list, input_y_list, lags, transpose):
    if type(input_x_list) == np.ndarray:
        input_x_list = [input_x_list]
    if type(input_y_list) == np.ndarray:
        input_y_list = [input_y_list]
    dataX, dataY = [], []
    for x, y in zip(input_x_list, input_y_list):
        print(len(x))
        temp_x, temp_y = create_samples_xy_rnn(x, y, lags, transpose)
        dataX.append(temp_x)
        dataY.append(temp_y)
    return flatten_list_3d(dataX), flatten_list(dataY)
        

def create_samples_x_rnn(dataset, lags):
    dataX, dataY = [], []
    for i in range(len(dataset) - lags):
        sample_X = dataset[i:i+lags, :]
        sample_Y = dataset[i+lags-1, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, D)
    dataY = np.array(dataY)  # (N, D)
    return dataX, dataY

class dataformat_for_rnn(Dataset):
    def __init__(self, train_x, train_y):
        self.X = train_x
        self.y = train_y

    def __getitem__(self, item):
        x_t = self.X[item]
        y_t = self.y[item]
        return x_t, y_t

    def __len__(self):
        return len(self.X)


# cc,dd=create_samples_xy_rnn(all_spike[0],all_emg[0],10,1)
# dataset = dataformat_for_rnn(cc, dd)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=3200, shuffle=True, sampler=None, batch_sampler=None)

# for x, y in dataloader:
#     print(x.shape)
#     print(y.shape)


