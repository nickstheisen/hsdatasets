#!/usr/bin/env python

from hsdatasets.groundbased.hsidrive import HSIDrive
#from hsdatasets.groundbased.hsidrive import HSIDriveDataset

if __name__ == '__main__':
    basepath = "/mnt/data/data/hsi-drive/Image_dataset"
    data_module = HSIDrive(
            basepath=basepath,
            train_prop=0.6,
            val_prop=0.2,
            batch_size=1, 
            num_workers=1)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    
    for i_batch, sample in enumerate(dataloader):
        data, label = sample
        print(f'{i_batch}: {data.shape}\t{label.shape}')
