#!/usr/bin/env python

import h5py

from typing import List, Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np

from hsdatasets.transforms import ToTensor, InsertEmptyChannelDim, PermuteData, ReplaceLabel, ReplaceLabels

class HSDataModule(pl.LightningDataModule):

    def __init__(
            self,
            filepath: str,
            num_workers: int = 8,
            batch_size: int = 8,
            train_prop: float = 0.7, # train proportion (of all data)
            val_prop: float = 0.1, # validation proportion (of all data)
    ):
        super().__init__()
        self.filepath = filepath
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.val_prop = val_prop

    def prepare_data(self):
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

class HyKo2(HSDataModule):
    def __init__(
            self,
            filepath: str,
            num_workers: int = 8,
            batch_size: int = 8,
            train_prop: float = 0.7, # train proportion (of all data)
            val_prop: float = 0.1, # validation proportion (of all data)
    ):
        super().__init__(
                filepath=filepath,
                num_workers=num_workers,
                batch_size=batch_size,
                train_prop=train_prop,
                val_prop=val_prop)

        self.transform = transforms.Compose([
            ToTensor(),
            PermuteData(new_order=[2,0,1]),
            ReplaceLabels({0:10, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}) # replace undefined label 0 with 10 and then shift labels by one
        ])


class HyperspectralCity2(HSDataModule):

    def __init__(
            self,
            filepath: str,
            num_workers: int = 8,
            batch_size: int = 8,
            train_prop: float = 0.7, # train proportion (of all data)
            val_prop: float = 0.1, # validation proportion (of all data)
    ):
        super().__init__(
                filepath=filepath,
                num_workers=num_workers,
                batch_size=batch_size,
                train_prop=train_prop,
                val_prop=val_prop)

        self.transform = transforms.Compose([
            ToTensor(),
            PermuteData(new_order=[2,0,1]),
            ReplaceLabel(255,19)
        ])

class GroundBasedHSDataset(Dataset):

    def __init__(self, filepath, transform):
        self._filepath = filepath
        
        # if h5file is kept open, the object cannot be pickled and in turn 
        # multi-gpu cannot be used
        h5file = h5py.File(self._filepath, 'r')
        self._samplelist = list(h5file.keys())
        self._transform = transform
        h5file.close()

    def __len__(self):
        return len(self._samplelist)

    def __getitem__(self, idx):
        h5file = h5py.File(self._filepath)
        sample = (np.array(h5file[self._samplelist[idx]]['data']),
                np.array(h5file[self._samplelist[idx]]['labels']))

        if self._transform:
            sample = self._transform(sample)
        h5file.close()
        return sample
