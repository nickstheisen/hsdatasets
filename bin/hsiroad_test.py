#!/usr/bin/env python

from hsdatasets.groundbased.groundbased import HSIRoad

if __name__ == '__main__':
    data_module = HSIRoad(
            basepath="/home/hyperseg/data/hsi_road/hsi_road",
            sensortype="nir",
            batch_size=1, 
            num_workers=1)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    
    for i_batch, sample in enumerate(dataloader):
        data, label = sample
        print(f'{i_batch}: {data.shape}\t{label.shape}')
