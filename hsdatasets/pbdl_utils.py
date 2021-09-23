#!/usr/bin/env python
import struct
import numpy as np
import cv2
import glob
import os
from pathlib import Path
import h5py
import argparse


def load_data_from_hsd(filename):

    with open(str(filename), 'rb') as f:

        # load meta infos
        height = struct.unpack('i', f.read(4))[0]
        width = struct.unpack('i', f.read(4))[0]
        bands = struct.unpack('i', f.read(4))[0]
        D = struct.unpack('i', f.read(4))[0]
        startw = struct.unpack('i', f.read(4))[0]
        stepw = struct.unpack('f', f.read(4))[0]
        endw = struct.unpack('i', f.read(4))[0]

        # load average values per band?
        averages = np.zeros((bands))
        for i in range(bands):
            averages[i] = struct.unpack('f', f.read(4))[0]
            
        # load coefficients for dimensionality reduction matrix
        coeffs = np.zeros((D*bands))
        for i in range(D*bands):
            coeffs[i] = struct.unpack('f', f.read(4))[0]
        
        # load dimensionality reduced data
        scoredata = np.zeros((height*width*D))
        for i in range(height*width*D):
            scoredata[i] = struct.unpack('f', f.read(4))[0]

        # reconstruct data
        coeffs = coeffs.reshape((D, bands), order='C')
        scoredata = scoredata.reshape((height*width, D), order='C')
        temp = scoredata @ coeffs
        data1 = temp + averages
        
        # reconstruct original structure of hyperspectral cube
        data = data1.reshape(height, width, bands, order='C')
        return data

def load_labels_from_png(filename):
    label_img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
    return label_img

def convert_hsds_to_hdf(input_dir, outfile):
    with h5py.File(outfile, "w") as hdf_file:
        # iterate over all hsd-files
        for file in input_dir.iterdir():
            if file.suffix == '.hsd':
                labelfile = input_dir.joinpath(f'rgb{file.stem}_gray.png')
                data = load_data_from_hsd(file)
                labels = load_labels_from_png(labelfile)
                
                # create group for image + labels
                group = hdf_file.create_group(file.name)
                group.create_dataset("data", data=data)
                group.create_dataset("labels", data=labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PBDL data set consisting of .hsd-files'
            ' to single hdf5-file.')
    parser.add_argument('input_dir', type=str, help='directory with hsd-files')
    parser.add_argument('output_filepath', type=str, help='filepath to hdf5-file')
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_filepath = Path(args.output_filepath).expanduser().resolve()
    #output_dir = output_filepath.parent
    #output_file = output_filepath.name

    convert_hsds_to_hdf(input_dir, output_filepath)
