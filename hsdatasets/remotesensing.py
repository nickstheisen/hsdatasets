#!/usr/bin/env python

import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
from urllib.request import urlretrieve
import numpy as np
from sklearn.decomposition import PCA
from hsdatasets.utils import TqdmUpTo

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
    def __init__(self, scene, root_dir='~/data',
                 window_size=1,
                 apply_pca=False,
                 pca_dim=75,
                 rm_zero_labels=True,
                 padding_mode='constant',
                 padding_values=0,
                 secure_sampling=True,
                 transform=None):
        # dataset config
        self.root_dir = Path(root_dir).expanduser()
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

        if apply_pca:
            data = self._apply_pca(data)

        if self.secure_sampling:
            self.data_patches, self.patch_labels = self._secure_sample_patches(
                    data, labels)
        else :
            self.data_patches, self.patch_labels = self._sample_patches(
                    data, labels,
                    self.window_size,
                    rm_zero_labels,
                    self.padding_mode,
                    self.padding_values)

    def __len__(self):
        return self.data_patches.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch = self.data_patches[idx]
        patch = np.array([patch])
        patch = patch.astype('float').transpose(0,3,2,1)

        sample = (patch, np.array(self.patch_labels[idx]))

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_data(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        filepath_data = self.root_dir.joinpath(DATASETS_CONFIG[self.scene]['img']['name'])
        filepath_labels = self.root_dir.joinpath(DATASETS_CONFIG[self.scene]['gt']['name'])

        if not filepath_data.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(filepath_data)) as t:
                url = DATASETS_CONFIG[self.scene]['img']['url']
                urlretrieve(url, filename=filepath_data, reporthook=t.update_to)

        if not filepath_labels.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(filepath_data)) as t:
                url = DATASETS_CONFIG[self.scene]['gt']['url']
                urlretrieve(url, filename=filepath_labels, reporthook=t.update_to)

        # dataset specific addressing
        if self.scene == 'IP':
            data = loadmat(filepath_data)['indian_pines_corrected']
            labels = loadmat(filepath_labels)['indian_pines_gt']
        elif self.scene == 'Salinas':
            data = loadmat(filepath_data)['salinas_corrected']
            labels = loadmat(filepath_labels)['salinas_gt']
        elif self.scene == 'SalinasA':
            data = loadmat(filepath_data)['salinasA_corrected']
            labels = loadmat(filepath_labels)['salinasA_gt']
        elif self.scene == 'PU':
            data = loadmat(filepath_data)['paviaU']
            labels = loadmat(filepath_labels)['paviaU_gt']
        elif self.scene == 'PC':
            data = loadmat(filepath_data)['pavia']
            labels = loadmat(filepath_labels)['pavia_gt']
        elif self.scene == 'KSC':
            data = loadmat(filepath_data)['KSC']
            labels = loadmat(filepath_labels)['KSC_gt']
        elif self.scene == 'Botswana':
            data = loadmat(filepath_data)['Botswana']
            labels = loadmat(filepath_labels)['Botswana_gt']
        return data, labels

    def _apply_pca(self, data):
        X = np.reshape(data, (-1, data.shape[2]))
        self.pca = PCA(n_components=self.pca_dim, whiten=True)
        X = self.pca.fit_transform(X)
        X = np.reshape(X, (data.shape[0], data.shape[1], self.pca_dim))
        return X

    def _sample_patches(self, data, labels):

        # pad along spatial axes
        margin = int((self.window_size - 1) / 2)
        X = np.pad(data, ((margin, margin), (margin, margin), (0,0)), 
                mode=self.padding_mode, constant_values=self.padding_values) 
        # split patches
        data_patches = np.zeros((data.shape[0] * data.shape[1], self.window_size, self.window_size, 
            data.shape[2]))
        patch_labels = np.zeros((data.shape[0] * data.shape[1]))
        patch_idx = 0
        for r in range(margin, X.shape[0] - margin):
            for c in range(margin, X.shape[1] - margin):
                patch = X[r - margin:r + margin + 1, c - margin:c + margin + 1]
                data_patches[patch_idx, :, :, :] = patch
                patch_labels[patch_idx] = labels[r-margin, c-margin]
                patch_idx += 1
        if self.rm_zero_labels:
            data_patches = data_patches[patch_labels>0,:,:,:]
            patch_labels = patch_labels[patch_labels>0]
            patch_labels -= 1
        return data_patches, patch_labels

    def _secure_sample_patches(self, data, labels):
        print('Secure sampling is not implemented yet! Falling back to normal sampling.')
        return _sample_patches(data, labels)

    def pca(self):
        return self.pca

class PaviaU(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='PU', *args, **kwargs)

class PaviaC(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='PC', *args, **kwargs)

class SalinasScene(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='Salinas', *args, **kwargs)

class SalinasA(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='SalinasA', *args, **kwargs)

class IndianPines(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='IP', *args, **kwargs)

class KennedySpaceCenter(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='KSC', *args, **kwargs)

class Botswana(HyperspectralDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(scene='Botswana', *args, **kwargs)
