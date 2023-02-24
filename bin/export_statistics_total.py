#!/usr/bin/env python

from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.hsidrive import HSIDrive
from hsdatasets.groundbased.groundbased import HyperspectralCityV2, HSIRoad,HyKo2
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pathlib import Path
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    #filepath = Path('/mnt/data/HyperspectralCityV2_PCA25.h5')
    #filepath = download_dataset('~/data/', 'HyKo2-NIR_Material')
    #dataset = GroundBasedHSDataset(filepath, transform=[])
    #dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    manual_seed=42
    seed_everything(manual_seed, workers=True)
    train_proportion = 1 
    val_proportion = 0

    #data_module = HSIDrive(
    #        basepath = "/mnt/data/data/hsi-drive/Image_dataset",
    #        train_prop=train_proportion,
    #        val_prop=val_proportion,
    #        batch_size=32,
    #        num_workers=8)
    #data_module.setup()

    #batch_size = 4
    #n_classes = 19 # 20 - 1 because class 255/19 is undefined
    #n_channels = 1 # apply DR to reduce from 128 to X
    #ignore_index = 19
    #num_workers = 14
    #pca_out_filepath = f'/mnt/data/HyperspectralCityV2_PCA{n_channels}.h5'
    #dataset_filepath = pca_out_filepath
    #half_precision=True
    #data_module = HyperspectralCityV2(
    #        half_precision=half_precision,
    #        filepath=dataset_filepath, 
    #        num_workers=num_workers,
    #        batch_size=batch_size,
    #        train_prop=train_proportion,
    #        val_prop=val_proportion,
    #        n_classes=n_classes,
    #        manual_seed=manual_seed)

    #n_classes = 2
    #data_module = HSIRoad(
    #        basepath="/home/hyperseg/data/hsi_road/hsi_road",
    #        sensortype="nir",
    #        batch_size=32, 
    #        num_workers=8)
    #data_module.setup()
    #dataloader = data_module.train_dataloader()

    hyko2vissem_filepath = download_dataset('~/data','HyKo2-VIS_Semantic')
    n_classes= 10 
    data_module = HyKo2(
            filepath=hyko2vissem_filepath, 
            num_workers=8,
            batch_size=32,
            label_set='semantic',
            train_prop=train_proportion,
            val_prop=val_proportion,
            n_classes=n_classes,
            manual_seed=manual_seed)
    data_module.setup()

    dataloader = data_module.train_dataloader()

    classes = n_classes
    aggregator = np.zeros((classes), dtype=np.int64)
    print(aggregator)
    
    for sample in tqdm(dataloader, total=len(dataloader)):
        _, label = sample
        for c in range(classes):
            aggregator[c] += np.count_nonzero(label==c)
    print(aggregator)
    np.savetxt("statistics-hyko2vissem.txt", aggregator)

    
