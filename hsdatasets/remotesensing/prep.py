#!/usr/bin/env python3

from pathlib import Path
from hsdatasets.utils import TqdmUpTo
import warnings
import gdown
from urllib.request import urlretrieve
from scipy.io import loadmat, savemat
import numpy as np
from math import ceil

DATASETS_CONFIG = {
        'IP' : {
            'img': {
                'name': 'Indian_pines_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                'key' : 'indian_pines_corrected'
            },
            'gt': {
                'name' : 'Indian_pines_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
                'key' : 'indian_pines_gt'
            }
        },
        'PU' : {
            'img': {
                'name' : 'PaviaU.mat',
                'url'  : 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                'key'  : 'paviaU'
            },
            'gt': {
                'name' : 'PaviaU_gt',
                'url'  : 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                'key'  : 'paviaU_gt'
            }
        },
        'Salinas' : {
            'img': {
                'name': 'Salinas_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
                'key' : 'salinas_corrected'
            },
            'gt': {
                'name': 'Salinas_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
                'key' : 'salinas_gt'
            }
        },
        'SalinasA' : {
            'img': {
                'name': 'SalinasA_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/1/1a/SalinasA_corrected.mat',
                'key' : 'salinasA_corrected'
            },
            'gt': {
                'name': 'SalinasA_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat',
                'key' : 'salinasA_gt'
            }
        },
        'PC' : {
            'img': {
                'name': 'Pavia.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
                'key' : 'pavia'
            },
            'gt': {
                'name': 'Pavia_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat',
                'key' : 'pavia_gt'
            }
        },
        'KSC' : {
            'img': {
                'name': 'KSC.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                'key' : 'KSC'
            },
            'gt': {
                'name': 'KSC_gt.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat',
                'key' : 'KSC_gt'
            }
        },
        'Botswana' : {
            'img': {
                'name': 'Botswana.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                'key' : 'Botswana'
            },
            'gt': {
                'name': 'Botswana_gt.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat',
                'key' : 'Botswana_gt'
            }
        },
        'AeroRIT_reflectance' : {
            'img': {
                'name': 'image_hsi_reflectance.tif',
                'url' : 'https://drive.google.com/uc?id=1OC-g9RdMxtd2lzyrR5Wvd-ioSg-DLN-l'
            },
            'gt': {
                'name': 'image_labels.tif',
                'url' : 'https://drive.google.com/uc?id=1e7_x887BiCboCKGNYodm8czHoTHxT8ro'
            }
        },
        'AeroRIT_radiance' : {
            'img': {
                'name': 'image_hsi_radiance.tif',
                'url' : 'https://drive.google.com/uc?id=15dL65NZqJSTRQaZzI9Ai33pLWWiAPnoZ'
            },
            'gt': {
                'name': 'image_labels.tif',
                'url' : 'https://drive.google.com/uc?id=1e7_x887BiCboCKGNYodm8czHoTHxT8ro'
            }
        }
}
def check_already_sampled(patchdir, trainlist, testlist):
    """ Checks if data was already sampled and loads data if possible.
        
        Throws a warning if patchtdir already exists. To resample this directory must be
        deleted or moved. 

        Raises a RuntimeError if trainlist and testlist do not exists. This is also
        the case if previously another split than the one defined in the filenames was
        calculated. Resample to generate a new split. 

        Arguments:
        ----------
        patchdir : PosixPath
            Directory where data patches are stored in.
        trainlist : str
            Name of file where train data paths where exported to.
        testlist : str
            Name of file where test data paths where exported to
    """
    if patchdir.is_dir():
        warnings.warn('Sampled data already exist, remove directory'
                ' "{}" to resample data!'.format(patchdir))
        trainsamplelist = patchdir.joinpath(trainlist)
        testsamplelist = patchdir.joinpath(testlist)

        # if this split was not defined yet raise exception
        if not trainsamplelist.is_file():
            raise RuntimeError('The given test-train split is not available.'
                ' To generate a new one please remove directory "{}" and resample '
                ' data!'.format(patchdir))

        trainsamples = np.array([
                Path(p) for p in np.loadtxt(trainsamplelist, dtype=str)
            ])
        testsamples = np.array([
                Path(p) for p in np.loadtxt(testsamplelist, dtype=str)
            ])
        return trainsamples, testsamples
    return None

def download_dataset(base_dir, scene):

    base_dir = Path(base_dir).expanduser().joinpath(scene)
    base_dir.mkdir(parents=True, exist_ok=True)
    filepath_data = base_dir.joinpath(DATASETS_CONFIG[scene]['img']['name'])
    filepath_labels = base_dir.joinpath(DATASETS_CONFIG[scene]['gt']['name'])

    if filepath_data.suffix == '.mat': # datasets from ehu.es
        if not filepath_data.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(filepath_data)) as t:
                url = DATASETS_CONFIG[scene]['img']['url']
                urlretrieve(url, filename=filepath_data, reporthook=t.update_to)

        if not filepath_labels.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(filepath_labels)) as t:
                url = DATASETS_CONFIG[scene]['gt']['url']
                urlretrieve(url, filename=filepath_labels, reporthook=t.update_to)
    elif filepath_data.suffix == '.tif': # aerorit
        if not filepath_data.is_file():
            print("Downloading {}".format(filepath_data))
            url = DATASETS_CONFIG[scene]['img']['url']
            gdown.download(url=url, output=str(filepath_data), quiet=False)

        if not filepath_labels.is_file():
            print("Downloading {}".format(filepath_labels))
            url = DATASETS_CONFIG[scene]['gt']['url']
            gdown.download(url=url, output=str(filepath_labels), quiet=False)

    return filepath_data, filepath_labels

def load_data(datapath, labelpath, scene):
    if datapath.suffix == '.mat':
        data = loadmat(datapath)[DATASETS_CONFIG[scene]['img']['key']]
        labels = loadmat(labelpath)[DATASETS_CONFIG[scene]['gt']['key']]
    elif datapath.suffix == '.tif':
        data = None
        labels = None
        print('tif-loading not implemented yet! Returning None')
    else :
        raise RuntimeError('Unknown filetype.')
    return data, labels

def _sample_patches(imgs, 
        labelimgs, 
        patch_size, 
        patchdir, 
        padding_mode, 
        padding_values, 
        ignore_labels,
        startpatchidx=0):
    """
    Samples datapatches from data and stores them in permanent memory. Returns paths to samples.

    data: list
        List of data to be sampled.
    labels: list
        List of labels to be sampled.
    startpatchidx: int, optional
        Patchindex to start counting from. To avoid overwriting data in 
    """
    samples = []
    for patchidx, (img, labelimg) in enumerate(zip(imgs, labelimgs)):

        # pad along spatial axes
        margin = int((patch_size - 1) / 2)
        X = np.pad(img, ((margin, margin), (margin, margin), (0,0)), 
                mode=padding_mode, constant_values=padding_values) 

        # split patches
        for r in range(margin, X.shape[0] - margin):
            for c in range(margin, X.shape[1] - margin):
                patchlabel = labelimg[r-margin, c-margin]

                # do not create a sample for 'ignore_labels'
                if patchlabel in ignore_labels:
                    continue
                else :
                    patchlabel -= 1

                patch = X[r - margin:r + margin + 1, c - margin:c + margin + 1]
                
                # store sample in permanent memory
                samplepath = patchdir.joinpath(f'p{startpatchidx + patchidx}_x{r}_y{c}.mat')
                samples.append(samplepath)
                sample = {'data': patch, 'label': patchlabel}
                savemat(samplepath, sample)
    return np.array(samples)

def split_random_sampling(datapath, labelpath, 
        patch_size, 
        train_ratio,
        outpath,
        padding_mode='constant', 
        padding_values=0, 
        ignore_labels=[0]):
        """
        Randomly samples data patches at each position and assigns them to test and train set.
        
        The hyperspectral data cube is padded along its spatial dimensions. Then
        a sample patch is extracted at each position. Each patch is stored as a mat-file
        together with its corresponding label in <root_dir>/<scene>/patches/. The data paths are
        stored in the same directory with name 'sample_list.txt'.

        If sampled data already exists their paths are simply imported from 'sample_list.txt'-file.
        To resample delete <root_dir>/<scene>/patches or move to another location.

        If no train-test-split is defined for given test-train-ratio it is created. Path to samples
        of train set are exported to '<root_dir>/<scene>/patches/train_<train_ratio>.txt' and 
        samples of test set to '<root_dir>/<scene>/patches/test_<(1-train_ratio)>.txt'

        If 'self.rm_zero_labels' is true, data with zero labels are ignored and all labels >=1 are 
        decremented.

        Attributes
        ----------

        data : ndarray
            Hyperspectral image from which patches are sampled.
        labels : ndarray
            Label image corresponding to hyperspectral image cube.
        """
        patchdir = outpath.joinpath('patches').joinpath('random_sampling')
        trainlist = f'train_{train_ratio:.2f}.txt'
        testlist = f'test_{1-train_ratio:.2f}.txt'
        
        samples = check_already_sampled(patchdir, trainlist, testlist)
        if samples is not None:
            return samples

        patchdir.mkdir(parents=True, exist_ok=True)
        
        data, labels = load_data(datapath, labelpath, outpath.name)
        # sample patches
        samples = _sample_patches([data], [labels], 
                    patch_size, 
                    patchdir, 
                    padding_mode, 
                    padding_values, 
                    ignore_labels)        

        # define test train split
        split_idx = (int) (train_ratio * len(samples))
        np.random.shuffle(samples)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        #self.samples = train_samples if self.train else test_samples
        
        # export test train split
        np.savetxt(patchdir.joinpath(trainlist), train_samples, fmt="%s")
        np.savetxt(patchdir.joinpath(testlist), test_samples, fmt="%s")
        
        return train_samples, test_samples


def split_secure_sampling(datapath, labelpath, 
        patch_size, 
        train_ratio,
        outpath,
        padding_mode='constant', 
        padding_values=0, 
        ignore_labels=[0]):
        """
        Samples a data patch at each position without overlap between test and training data.

        If sampled data already exists their paths are simply imported from 'sample_list.txt'-file.
        To resample delete <root_dir>/<scene>/patches or move to another location.

        If no train-test-split is defined for given test-train-ratio it is created. Path to samples
        of train set are exported to '<root_dir>/<scene>/patches/train_<train_ratio>.txt' and 
        samples of test set to '<root_dir>/<scene>/patches/test_<(1-train_ratio)>.txt'

        If 'self.rm_zero_labels' is true, data with zero labels are ignored and all labels >=1 are 
        decremented.

        Attributes
        ----------

        data : ndarray
            Hyperspectral image from which patches are sampled.
        labels : ndarray
            Label image corresponding to hyperspectral image cube.

        """

        # directory to store samples in
        patchdir = outpath.joinpath('patches').joinpath('secure_sampling')
        trainlist = f'train_{train_ratio:.2f}.txt'
        testlist = f'test_{1-train_ratio:.2f}.txt'

        samples = check_already_sampled(patchdir, trainlist, testlist)
        if samples is not None:
            return samples

        data, labels = load_data(datapath, labelpath, outpath.name)

        patchdir.mkdir(parents=True, exist_ok=True)

        # split image into subimages of size patch_size x patch_size
        h, w, _ = data.shape
        num_subimg_h = ceil(h/patch_size) # patches along vertical axis
        num_subimg_w = ceil(w/patch_size) # patches along horizontal axis

        subimgs = []
        subimg_labels = []

        for i in range(num_subimg_h):
            for j in range(num_subimg_w):
                start_idx_h = i*patch_size
                start_idx_w = j*patch_size
                end_idx_h = (i+1)*patch_size
                end_idx_w = (j+1)*patch_size

                # end_idx_h and end_idx_w may be greater than height and width of data array
                if end_idx_h > h:
                    end_idx_h = h
                if end_idx_w > w:
                    end_idx_w = w

                subimgs.append(data[start_idx_h:end_idx_h, start_idx_w:end_idx_w])
                subimg_labels.append(labels[start_idx_h:end_idx_h, start_idx_w:end_idx_w])

        # shuffle samples
        samples = list(zip(subimgs, subimg_labels))
        np.random.shuffle(samples)
        subimgs, subimg_labels = zip(*samples)

        # count how many pixels have non 'ignore_labels' and use result to assign approximately
        # train_ratio share of non zero data to train set and (1-train_ratio) to test set
        if ignore_labels:
            cum_nonzero_labels = np.cumsum(
                    [np.isin(lbls, ignore_labels).sum() for lbls in subimg_labels])
            split = 0
            if cum_nonzero_labels[-1] == 0:
                raise RuntimeError('Labelimage only contains zero labels.')
            print(f'{cum_nonzero_labels[split]} / {cum_nonzero_labels[-1]}')
            while(cum_nonzero_labels[split]/cum_nonzero_labels[-1] < train_ratio):
                split += 1
        else :
            split = int((len(subimgs)*train_ratio))

        # sample test and training data patches
        train_data = subimgs[:split]
        train_label_patches = subimg_labels[:split]
        test_data = subimgs[split:]
        test_label_patches = subimg_labels[split:]
        train_samples = _sample_patches(train_data, train_label_patches, 
                    patch_size, 
                    patchdir, 
                    padding_mode, 
                    padding_values, 
                    ignore_labels)
        test_samples = _sample_patches(test_data, test_label_patches, 
                    patch_size, 
                    patchdir, 
                    padding_mode, 
                    padding_values, 
                    ignore_labels,
                    startpatchidx=len(train_data))

        #samples = train_samples if self.train else test_samples
        
        # export test train split
        np.savetxt(patchdir.joinpath(trainlist), train_samples, fmt="%s")
        np.savetxt(patchdir.joinpath(testlist), test_samples, fmt="%s")

        return train_samples, test_samples


