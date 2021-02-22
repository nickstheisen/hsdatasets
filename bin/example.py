#!/usr/bin/env python

from hsdatasets.remotesensing import (  IndianPines, PaviaU, SalinasScene, PaviaC, 
                                        SalinasA, KennedySpaceCenter, Botswana )


if __name__ == '__main__':
    ip = IndianPines(train=True,
            window_size=25,
            pca_dim=30,
            train_ratio=0.8,
            secure_sampling=False)
