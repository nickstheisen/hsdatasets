#!/usr/bin/env python

import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat, savemat
import numpy as np
from sklearn.decomposition import PCA
import warnings
from math import ceil

import h5py

class RemoteSensingDataset(Dataset):
    """
    Base class to represent hyperspectral remote sensing data sets.

    Attributes
    ----------
    filepath : PosixPath
        Path to file containing train and test data (.h5, .hdf5).
    train : bool
        Defines if train or test data is returned.
    apply_pca : bool
        Defines if dimensionality is reduced with PCA.
    pca_dim : int
        Number of dimensions that data is reduced to.
    transform : functional
        Transformations applied to data samples.
    pca : object
        Stores sklearn.decomposition.PCA object that is used to reduced dimensionality
    train : bool
        True to get training set, False to get test set.

    Methods
    -------

    __len__():
        Return length of data set.

    __getitem__(self, idx):
        Return sample in data set at position 'idx'.

    _pca(self, data):
        Applies PCA to reduce datas spectral dimensionality.

    """
    def __init__(self, filepath, train,
                 apply_pca=False,
                 pca_dim=75,
                 transform=None):

        self._file = h5py.File(filepath, 'r')
        self._samplelist = self._file['trainsample_list'] if train else self._file['testsample_list']

        self._transform = transform
        self._apply_pca = apply_pca
        self._pca_dim = pca_dim

        self._train = train
        
        if apply_pca:
            self._pca = self.calc_pca(self._file['patches']['data'], self._pca_dim)

    def __len__(self):
        """ Return length of data set."""
        return self._samplelist.shape[0]

    def __getitem__(self, idx):
        """ 
        Return sample in data set at position 'idx'.

        Data patch and corresponding are loaded from hard disk. Transformations defined in
        `self._transform` are applied before returning the samples.

        Attributes:
        -----------
        idx : int
            Position of data in sample list.
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pos = self._samplelist[idx]

        patch = self._file['patches']['data'][pos]
        label = self._file['patches']['labels'][pos]
        
        # apply pca
        if self._apply_pca:
            shape = patch.shape
            patch = np.reshape(patch, (-1, patch.shape[2]))
            patch = self.pca.transform(patch)
            patch = np.reshape(patch, (shape[0], shape[1], self._pca_dim))
    
        sample = (patch, label)

        if self._transform:
            sample = self._transform(sample)

        return sample

    @staticmethod
    def calc_pca(patches, dim):
        """
        Calculates PCA to reduce the data's spectral dimensionality. Resulting PCA-object is
        returned. sklearn.decomposition.PCA is applied to data to reduce dimensionality. 
        """

        # get all samples
        X = []
        shape = patches[0].shape
        cx, cy = shape[0]//2, shape[1]//2
        for patch in patches:
           # because of patch sampling only center pixel is relevant
           # all other pixels are padding
           X.append(patch[cx,cy,:])
        
        # fit pca
        X = np.array(X)
        pca = PCA(n_components=dim, whiten=True)
        pca.fit(X)
        return pca
    
    @property
    def pca(self):
        """
        Returns pca object or None of if it does not exist.
        """
        return self._pca

    @pca.setter
    def pca(self, value):
        """
        Set pca Object.
        """
        self._pca = value
        if value is None:
            self._apply_pca = False
            self._pca_dim = None
        else : 
            self._apply_pca = True
            self._pca_dim = value.n_components
