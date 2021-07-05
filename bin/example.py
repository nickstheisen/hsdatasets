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
    #scene = 'AeroRIT_radiance_mid'
    scene = 'SalinasA'

    # download data if not already existing
    dataset_path = download_dataset(base_dir='~/data', scene=scene)
    
    # sample data and split into test and trainset
    filepath = split_secure_sampling(dataset_path, 7, 0.7, dataset_path.parents[0])

    # create dataset
    transform = transforms.Compose([ToTensor()])
    salinasa_train = RemoteSensingDataset(filepath, train=True, apply_pca=True, pca_dim=10, transform=transform)
    # create dataloader
    salinasa_trainloader = torch.utils.data.DataLoader(salinasa_train,
            batch_size=4, shuffle=True, num_workers=2)

    for i, data in enumerate(salinasa_trainloader, 0):
        inputs, labels = data
        print(i, inputs.shape)
