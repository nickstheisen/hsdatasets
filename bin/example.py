#!/usr/bin/env python

#from hsdatasets.remotesensing import (  IndianPines, PaviaU, SalinasScene, PaviaC, 
#                                        SalinasA, KennedySpaceCenter, Botswana )
from pathlib import Path

from hsdatasets.remotesensing.prep import download_dataset, split_random_sampling, split_secure_sampling

if __name__ == '__main__':
    filepath_data, filepath_labels = download_dataset(base_dir='~/data', scene='SalinasA')

    print(split_secure_sampling(filepath_data, filepath_labels, 25, 0.7, filepath_data.parents[0]))

#    ip = IndianPines(train=True,
#            window_size=25,
#            pca_dim=30,
#            train_ratio=0.8,
#            secure_sampling=False)
