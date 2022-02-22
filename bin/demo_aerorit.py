#!/usr/bin/env python

import h5py
from hsdatasets.remotesensing.prep import download_dataset
from hsdatasets.transforms import ToTensor

if __name__ == '__main__':

    scene_train = 'AeroRIT_radiance_left'
    scene_valid = 'AeroRIT_radiance_mid'
    scene_test = 'AeroRIT_radiance_right'

    dataset_path_train = download_dataset(base_dir='/tmp', scene=scene_train)
    dataset_path_valid = download_dataset(base_dir='/tmp', scene=scene_valid)
    dataset_path_test = download_dataset(base_dir='/tmp', scene=scene_test)

