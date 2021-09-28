#!/usr/bin/env python

import h5py

from typing import List, Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np

from hsdatasets.transforms import ToTensor, InsertEmptyChannelDim, PermuteData

class HSDataModule(pl.LightningDataModule):

    def __init__(
            self,
            filepath: str,
            num_workers: int = 8,
            batch_size: int = 8,
            train_prop: float = 0.7, # train proportion (of all data)
            val_prop: float = 0.1, # validation proportion (of all data)
            pca_dim: int = 75,
    ):
        super().__init__()
        self.filepath = filepath
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.pca_dim = pca_dim

    def prepare_data(self):
        self.transform = transforms.Compose([
            ToTensor(),
            PermuteData(new_order=[2,0,1])])

        dataset = GroundBasedHSDataset(self.filepath, transform=self.transform)
        train_size = round(self.train_prop * len(dataset))
        val_size = round(self.val_prop * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)


class GroundBasedHSDataset(Dataset):

    def __init__(self, filepath, transform):
        self._file = h5py.File(filepath, 'r')
        self._samplelist = list(self._file.keys())
        self._transform = transform

    def __len__(self):
        return len(self._samplelist)

    def __getitem__(self, idx):
        sample = (np.array(self._file[self._samplelist[idx]]['data']),
                np.array(self._file[self._samplelist[idx]]['labels']))

        if self._transform:
            sample = self._transform(sample)
        return sample
