#!/usr/bin/env python

#from hsdatasets.remotesensing import (  IndianPines, PaviaU, SalinasScene, PaviaC, 
#                                        SalinasA, KennedySpaceCenter, Botswana )
from pathlib import Path

from hsdatasets.remotesensing.prep import download_dataset, split_random_sampling, split_secure_sampling
from hsdatasets.remotesensing import RemoteSensingDataset
from hsdatasets.transforms import ToTensor

import torch
from torchvision import transforms

if __name__ == '__main__':
    # download data if not already existing
    filepath_data, filepath_labels = download_dataset(base_dir='~/data', scene='SalinasA')

    # sample data and split into test and trainset
    train_list, test_list = split_secure_sampling(
            filepath_data, filepath_labels, 25, 0.7, filepath_data.parents[0])

    # create dataset
    transform = transforms.Compose([ToTensor()])
    salinasa_train = RemoteSensingDataset(train=True, samplelist=train_list, apply_pca=True, pca_dim=75, transform=transform)

    # create dataloader
    salinasa_trainloader = torch.utils.data.DataLoader(salinasa_train,
            batch_size=4, shuffle=True, num_workers=2)

    for i, data in enumerate(salinasa_trainloader, 0):
        inputs, labels = data
        print(i, inputs.shape)
