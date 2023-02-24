#!/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
from hsdatasets.transforms import ToTensor, ReplaceLabels

from typing import List, Any, Optional
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from imageio import imread

def label_histogram(dataset, n_classes):
    label_hist = torch.zeros(n_classes) # do not count 'unefined'(highest class_id)
    for i, (_, labels) in enumerate(DataLoader(dataset)):
        label_ids, counts = labels.unique(return_counts=True)
        for i in range(len(label_ids)):
            label_id = label_ids[i]
            if not (label_id == n_classes):
                label_hist[label_id] += counts[i]
    return label_hist

class HSIDriveDataset(Dataset):

    def __init__(self, basepath, transform):
        self._basepath = Path(basepath)
        self._labelpath = self._basepath.joinpath('labels')
        self._datapath = self._basepath.joinpath('cubes_fl32')
        self._samplelist = np.array(
            [sample.stem for sample in self._labelpath.glob('*.png')]
        )
        
        self._transform = transform

    def __len__(self):
        return len(self._samplelist)

    def samplelist(self):
        return self._samplelist

    def __getitem__(self, idx):
        samplename = self._samplelist[idx]
        data = loadmat(self._datapath.joinpath(samplename + "_TC.mat"))['cube']
        label = imread(self._labelpath.joinpath(samplename + ".png"))

        sample = (data, label)

        if self._transform:
            sample = self._transform(sample)
        return sample

class HSIDrive(pl.LightningDataModule):
    def __init__(
        self,
        basepath: str,
        batch_size: int,
        num_workers: int,
        train_prop: float, # train proportion (of all data)
        val_prop: float, # validation proportion (of all data)
        manual_seed: int=None,
        precalc_histograms: bool=False,
        ):
        super().__init__()
        
        self.save_hyperparameters()

        self.basepath = Path(basepath)
        self.train_prop = train_prop
        self.val_prop = val_prop
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
                            ToTensor(),
                            ReplaceLabels({0:10, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}) # replace undefined label 0 with 10 and then shift labels by one
                        ])
        self.manual_seed = manual_seed
        self.precalc_histograms=precalc_histograms
        self.c_hist_train = None
        self.c_hist_val = None
        self.c_hist_test = None

        self.n_classes = 10

    def class_histograms(self):
        if self.c_hist_train is not None :
            return (self.c_hist_train, self.c_hist_val, self.c_hist_test)
        else :
            return None

    def setup(self, stage: Optional[str] = None):
        dataset = HSIDriveDataset(self.basepath, self.transform)
        train_size = round(self.train_prop * len(dataset))
        val_size = round(self.val_prop * len(dataset))
        test_size = len(dataset) - (train_size + val_size)

        if self.manual_seed is not None:
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                    dataset, 
                    [train_size, val_size, test_size], 
                    generator=torch.Generator().manual_seed(self.manual_seed))
        else:
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                    dataset, 
                    [train_size, val_size, test_size])

        # calculate class_histograms
        if self.precalc_histograms:
            self.c_hist_train = label_histogram(
                    self.dataset_train, self.n_classes)
            self.c_hist_val = label_histogram(
                    self.dataset_val, self.n_classes)
            self.c_hist_test = label_histogram(
                    self.dataset_test, self.n_classes)


    def train_dataloader(self):
        return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

    def test_dataloader(self):
        # using val-set is not a typo, unfortunately the dataset authors provide only train and valid
        # ation sets. They use the validation set also as test set. We do the same to keep experiments
        # comparable
        return DataLoader(
                self.dataset_val, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)


