#!/usr/bin/env python

from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.groundbased import GroundBasedHSDataset
from torch.utils.data import DataLoader
from pathlib import Path

if __name__ == '__main__':
    hyko2vissem_filepath = Path('/data/nicktheisen/HyperspectralCityV2.0/HyperspectralCityV2.h5') #download_dataset('~/data/', 'HyKo2-VIS_Semantic')
    dataset = GroundBasedHSDataset(hyko2vissem_filepath)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    for i_batch, sample in enumerate(dataloader):
        data, label = sample
        print(f'{i_batch}: {data.shape}\t{label.shape}')
