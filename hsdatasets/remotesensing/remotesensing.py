#!/usr/bin/env python

import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat, savemat
import numpy as np
from sklearn.decomposition import PCA
import warnings
from math import ceil
from skimage import io

class RemoteSensingDataset(Dataset):
    """
    Base class to represent hyperspectral remote sensing data sets.

    Attributes
    ----------
    samplelist : PosixPath
        Path to list of presampled hyperspectral patches (.txt).
    scene : str
        Internal short name of data set. Used as name for directory where data set is stored in.
    transform : functional
        Transformations applied to data samples.
    apply_pca : bool
        Defines if dimensionality is reduced with PCA.
    pca_dim : int
        Number of dimensions that data is reduced to.
    samples : ndarray
        List of paths to data samples.
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
    def __init__(self, train, samplelist,
                 apply_pca=False,
                 pca_dim=75,
                 transform=None):
        """
        Instantiates hyperspectral dataset class.

        During instantiation the dataset is downloaded if necessary and data samples are 
        extracted and stored on in persistent memory as defined by 'root_dir' and 'scene'.

        Parameters
        ----------
        train : bool
            True to get training set, False to get test set.
        scene : str
            Internal short name of data set. Used as name for directory where data set is stored in.
        samplelist : PosixPath
            Path to list of presampled hyperspectral patches (.txt).
        apply_pca : bool, optional
            Defines if dimensionality is reduced with PCA (default=False).
        pca_dim : int, optional
            Number of dimensions that data is reduced to (default=75).
        transform : functional, optional
            Transformations applied to data samples. (default='None')
        """

        # dataset config
        self.samplelist = Path(samplelist).expanduser()

        # load list of patches from samplelist
        self.samples = np.array([
                    Path(p) for p in np.loadtxt(self.samplelist, dtype=str)
                    ])
        # data processing config
        self.transform = transform
        self.apply_pca = apply_pca
        self.pca_dim = pca_dim

        self.train = train

        if apply_pca:
            self.pca = self._pca(self.samples, self.pca_dim)

    def __len__(self):
        """ Return length of data set."""
        return self.samples.shape[0]

    def __getitem__(self, idx):
        """ 
        Return sample in data set at position 'idx'.

        Data patch and corresponding are loaded from hard disk. The path to it is taken
        from 'self.samples' list. Transformations defined by 'self.transform' are applied 
        before returning the samples.

        Attributes:
        -----------
        idx : int
            Position of data in sample list.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = loadmat(self.samples[idx])
        patch = np.array([sample['data']])

        # apply precomputed pca
        if self.pca:
            shape = patch.shape
            patch = np.reshape(patch, (-1, patch.shape[3]))
            patch = self.pca.transform(patch)
            patch = np.reshape(patch, (shape[0],shape[1], shape[2], self.pca_dim))
            
        label = sample['label'].squeeze()
        patch = patch.astype('float').transpose(0,3,2,1) # convert to NCHW-order

        sample = (patch, label)

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def _pca(samples, dim):
        """
        Applies PCA to reduce datas spectral dimensionality.

        sklearn.decomposition.PCA is applied to data to reduce dimensionality. 
        The object is stored in member variable to use e.g. for inverse transformation.
        """

        # get all samples
        X = []
        shape = loadmat(samples[0])['data'].shape
        cx, cy = shape[0]//2, shape[1]//2
        for sample in samples:
           # because of patch sampling only center pixel is relevant
           X.append(loadmat(sample)['data'][cx,cy,:])
        
        # fit pca
        X = np.array(X)
        print(X.shape)
        pca = PCA(n_components=dim, whiten=True)
        pca.fit(X)
        return pca

    def get_pca(self):
        return self.pca
