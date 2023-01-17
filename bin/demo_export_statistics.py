#!/usr/bin/env python

from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.groundbased import GroundBasedHSDataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    filepath = Path('/mnt/data/HyperspectralCityV2_PCA25.h5')
    #filepath = download_dataset('~/data/', 'HyKo2-NIR_Material')
    dataset = GroundBasedHSDataset(filepath, transform=[])
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    classes = 20
    aggregator = np.zeros((classes), dtype=np.int64)
    print(aggregator)
    
    for sample in tqdm(dataloader, total=len(dataloader)):
        _, label = sample
        for c in range(classes):
            aggregator[c] += np.count_nonzero(label==c)
    print(aggregator)
    np.savetxt("statistics-hcv2.txt", aggregator)
