#!/usr/bin/env python

import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat, savemat
from urllib.request import urlretrieve
import numpy as np
from sklearn.decomposition import PCA
from hsdatasets.utils import TqdmUpTo
from shutil import rmtree
import warnings

DATASETS_CONFIG = {
        'IP' : {
            'img': {
                'name': 'Indian_pines_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
            },
            'gt': {
                'name' : 'Indian_pines_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
            }
        },
        'PU' : {
            'img': {
                'name' : 'PaviaU.mat',
                'url'  : 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat'
            },
            'gt': {
                'name' : 'PaviaU_gt',
                'url'  : 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'
            }
        },
        'Salinas' : {
            'img': {
                'name': 'Salinas_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat'
            },
            'gt': {
                'name': 'Salinas_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat'
            }
        },
        'SalinasA' : {
            'img': {
                'name': 'SalinasA_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/1/1a/SalinasA_corrected.mat'
            },
            'gt': {
                'name': 'SalinasA_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat'
            }
        },
        'PC' : {
            'img': {
                'name': 'Pavia.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat'
            },
            'gt': {
                'name': 'Pavia_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'
            }
        },
        'KSC' : {
            'img': {
                'name': 'KSC.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat'
            },
            'gt': {
                'name': 'KSC_gt.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'
            }
        },
        'Botswana' : {
            'img': {
                'name': 'Botswana.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat'
            },
            'gt': {
                'name': 'Botswana_gt.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
            }
        }
}

class HyperspectralDataset(Dataset):
    """
    Base class to represent hyperspectral data sets.

    Attributes
    ----------
    root_dir : PosixPath
        Root directory where data sets are stored in.
    scene : str
        Internal short name of data set. Used as name for directory where data set is stored in.
    transform : functional
        Transformations applied to data samples.
    apply_pca : bool
        Defines if dimensionality is reduced with PCA.
    pca_dim : int
        Number of dimensions that data is reduced to.
    rm_zeros_labels : bool, optional
            All data with class-ID 0 is removed. Other class-IDs are decremented.
    window_size : int
        Sidelength of patches that are sampled from hyperspectral image.
    padding_mode : str or function
        Defines how data is padded (See numpy.pad) for further information.
    padding_values : sequence or scalar, optional
        Used with 'constant' padding mode. The values to set the padded values for each axis
        (default=0).

    secure_sampling : True
        Use secure sampling. No overlap between test and train data.
    samples : ndarray
        List of paths to data samples.
    pca : object
        Stores sklearn.decomposition.PCA object that was used to reduce dimensionality
    filepath_data : PosixPath
        Path to hyperspectral data.
    filepath_lables : PosixPath
        Path to labels corresponding to hyperspectral pixels.

    Methods
    -------

    __len__():
        Return length of data set.

    __getitem__(self, idx):
        Return sample in data set at position 'idx'.

    _load_data(self):
        Downloads data and labels if not existing.

    _apply_pca(self, data):
        Applies PCA to reduce datas spectral dimensionality.

     _sample_patches(self, data, labels):
        Samples a data patch at each pixel. 

    _secure_sample_patches(self, data, labels):
        Samples a data patch at each position without overlap between test and training data.
    """
    def __init__(self, scene, root_dir='~/data',
                 window_size=1,
                 apply_pca=False,
                 pca_dim=75,
                 rm_zero_labels=True,
                 padding_mode='constant',
                 padding_values=0,
                 secure_sampling=True,
                 transform=None):
        """
        Instantiates hyperspectral dataset class.

        During instantiation the dataset is downloaded if necessary and data samples are 
        extracted and stored on in persistent memory as defined by 'root_dir' and 'scene'.

        Parameters
        ----------
        scene : str
            Internal short name of data set. Used as name for directory where data set is stored in.
        root_dir : PosixPath, optional
            Root directory where data sets are stored in (default='~/data').
        window_size : int, optional
            Sidelength of patches that are sampled from hyperspectral image (default=1).
        apply_pca : bool, optional
            Defines if dimensionality is reduced with PCA (default=False).
        pca_dim : int, optional
            Number of dimensions that data is reduced to (default=75).
        rm_zeros_labels : bool, optional
            All data with class-ID 0 is removed. Other class-IDs are decremented (default=True).
        padding_mode : str or function, optional
            Defines how data is padded (See numpy.pad) for further information (default='constant').
        padding_values : sequence or scalar, optional
            Used with 'constant' padding mode. The values to set the padded values for each axis
            (default=0).
        secure_sampling : True, optional
            Use secure sampling. No overlap between test and train data (default=True).
        transform : functional, optional
            Transformations applied to data samples. (default='None')
        """

        # dataset config
        self.root_dir = Path(root_dir).expanduser().joinpath(scene)
        self.scene=scene

        # data processing config
        self.transform = transform
        self.apply_pca = apply_pca
        self.pca_dim = pca_dim

        # sampling config
        self.window_size = window_size
        self.rm_zero_labels = rm_zero_labels
        self.padding_mode=padding_mode
        self.padding_values=padding_values
        self.secure_sampling = secure_sampling

        # prepare data
        data, labels = self._load_data()
        self.samples = np.array([])

        if apply_pca:
            data = self._apply_pca(data)

        if self.secure_sampling:
            self._secure_sample_patches(
                    data, labels)
        else :
            self._sample_patches(
                    data, labels,
                    self.window_size,
                    rm_zero_labels,
                    self.padding_mode,
                    self.padding_values)

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
        label = sample['label'].squeeze()
        patch = patch.astype('float').transpose(0,3,2,1) # convert to NCHW-order

        sample = (patch, label)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_data(self):
        """
        Downloads data and labels if not existing.

        Creates directory and parent directories <root_dir>/<scene>. 
        Data set and label image is downloaded into those directories.
        """
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.filepath_data = self.root_dir.joinpath(DATASETS_CONFIG[self.scene]['img']['name'])
        self.filepath_labels = self.root_dir.joinpath(DATASETS_CONFIG[self.scene]['gt']['name'])

        if not self.filepath_data.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(self.filepath_data)) as t:
                url = DATASETS_CONFIG[self.scene]['img']['url']
                urlretrieve(url, filename=self.filepath_data, reporthook=t.update_to)

        if not self.filepath_labels.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(self.filepath_labels)) as t:
                url = DATASETS_CONFIG[self.scene]['gt']['url']
                urlretrieve(url, filename=self.filepath_labels, reporthook=t.update_to)

    def _apply_pca(self, data):
        """
        Applies PCA to reduce datas spectral dimensionality.

        sklearn.decomposition.PCA is applied to data to reduce dimensionality. 
        The object is stored in member variable to use e.g. for inverse transformation.
        """
        X = np.reshape(data, (-1, data.shape[2]))
        self.pca = PCA(n_components=self.pca_dim, whiten=True)
        X = self.pca.fit_transform(X)
        X = np.reshape(X, (data.shape[0], data.shape[1], self.pca_dim))
        return X

    def _sample_patches(self, data, labels):
        """
        Samples a data patch at each pixel. 
        
        The hyperspectral data cube is padded along its spatial dimensions. Then
        a sample patch is extracted at each position. Each patch is stored as a mat-file
        together with its corresponding label in <root_dir>/<scene>/patches/. The data paths are
        stored in the same directory with name 'sample_list.txt'.

        If sampled data already exists their paths are simply imported from 'sample_list.txt'-file.
        To resample delete <root_dir>/<scene>/patches or move to another location.

        If 'self.rm_zero_labels' is true, data with zero labels are ignored and all labels >=1 are 
        decremented.

        Attributes
        ----------

        data : ndarray
            Hyperspectral image from which patches are sampled.
        labels : ndarray
            Label image corresponding to hyperspectral image cube.
        """
        patchdir = self.root_dir.joinpath('patches')
        if patchdir.is_dir():
            warnings.warn('Sampled data already exists remove directory'
                    ' "{}" to resample data!'.format(patchdir))
            
            self.samples = np.array([
                    Path(p) for p in np.loadtxt(patchdir.joinpath('sample_list.txt'), dtype=str)
                ])
            return

        patchdir.mkdir(parents=True, exist_ok=True)

        # pad along spatial axes
        margin = int((self.window_size - 1) / 2)
        X = np.pad(data, ((margin, margin), (margin, margin), (0,0)), 
                mode=self.padding_mode, constant_values=self.padding_values) 

        # split patches
        samples = []
        for r in range(margin, X.shape[0] - margin):
            for c in range(margin, X.shape[1] - margin):
                patchlabel = labels[r-margin, c-margin]

                # if '0' label is removed '1' is new '0'
                if self.rm_zero_labels:
                    if patchlabel == 0:
                        continue
                    else :
                        patchlabel -= 1

                patch = X[r - margin:r + margin + 1, c - margin:c + margin + 1]
                
                # store sample in permanent memory
                samplepath = patchdir.joinpath(f'x{r}_y{c}.mat')
                samples.append(samplepath)
                sample = {'data': patch, 'label': patchlabel}
                savemat(samplepath, sample)
        
        self.samples = np.array(samples)

        np.savetxt(patchdir.joinpath('sample_list.txt'), self.samples, fmt="%s")

    def _secure_sample_patches(self, data, labels):
        """
        Samples a data patch at each position without overlap between test and training data.
        """
        warnings.warn('Secure sampling is not implemented yet! Falling back to normal sampling.')
        return self._sample_patches(data, labels)

    def pca(self):
        return self.pca

class PaviaU(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='PU', *args, **kwargs)

    def _load_data(self):
        super()._load_data()
        data = loadmat(self.filepath_data)['paviaU']
        labels = loadmat(self.filepath_labels)['paviaU_gt']
        return data, labels


class PaviaC(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='PC', *args, **kwargs)

    def _load_data(self):
        super()._load_data()
        data = loadmat(self.filepath_data)['pavia']
        labels = loadmat(self.filepath_labels)['pavia_gt']
        return data, labels


class SalinasScene(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='Salinas', *args, **kwargs)

    def _load_data(self):
        super()._load_data()
        data = loadmat(self.filepath_data)['salinas_corrected']
        labels = loadmat(self.filepath_labels)['salinas_gt']
        return data, labels


class SalinasA(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='SalinasA', *args, **kwargs)

    def _load_data(self):
        super()._load_data()
        data = loadmat(self.filepath_data)['salinasA_corrected']
        labels = loadmat(self.filepath_labels)['salinasA_gt']
        return data, labels


class IndianPines(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='IP', *args, **kwargs)

    def _load_data(self):
        super()._load_data()
        data = loadmat(self.filepath_data)['indian_pines_corrected']
        labels = loadmat(self.filepath_labels)['indian_pines_gt']
        return data, labels


class KennedySpaceCenter(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='KSC', *args, **kwargs)

    def _load_data(self):
        super()._load_data()
        data = loadmat(self.filepath_data)['KSC']
        labels = loadmat(self.filepath_labels)['KSC_gt']
        return data, labels

class Botswana(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='Botswana', *args, **kwargs)

    def _load_data(self):
        super()._load_data()
        data = loadmat(self.filepath_data)['Botswana']
        labels = loadmat(self.filepath_labels)['Botswana_gt']
        return data, labels

