#!/usr/bin/env python

import h5py

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np

class GroundBasedHSDataset(Dataset):

    def __init__(self, filepath):
        self._file = h5py.File(filepath, 'r')
        self._samplelist = list(self._file.keys())

    def __len__(self):
        return len(self._samplelist)

    def __getitem__(self, idx):
        sample = (np.array(self._file[self._samplelist[idx]]['data']),
                np.array(self._file[self._samplelist[idx]]['labels']))
        return sample
