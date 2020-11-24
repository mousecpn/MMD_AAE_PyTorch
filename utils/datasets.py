import torch
from torch.utils.data import Dataset
import scipy.io as sp
import os
import numpy as np

class VLSC_dataset(Dataset):
    def __init__(self, data,labels):
        super(VLSC_dataset,self).__init__()
        self.data = data.copy()
        self.labels = labels.copy()

        # self.cal_mean_std()


    def cal_mean_std(self):
        self.mean = np.mean(self.data, axis = 0)
        self.std = np.std(self.data, axis = 0)
        return

    def set_mean_std(self,mean,std):
        self.mean = mean
        self.std = std
        return

    def __getitem__(self, index):
        sample = self.data[index].copy()
        label = self.labels[index].copy()

        # normalize
        sample -= self.mean
        sample /= self.std

        sample = torch.tensor(sample)
        label = torch.tensor(int(label))

        return sample,label

    def __len__(self):
        return self.data.shape[0]


