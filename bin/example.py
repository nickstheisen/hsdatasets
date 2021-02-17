#!/usr/bin/env python

from hsdatasets.remotesensing import (  IndianPines, PaviaU, SalinasScene, PaviaC, 
                                        SalinasA, KennedySpaceCenter, Botswana )


if __name__ == '__main__':

    # load datasets
    #indian_pines = IndianPines()
    pavia_u = PaviaU(apply_pca=False)
    print(pavia_u[0])
    #salinas = SalinasScene()
    #salinasa = SalinasA()
    #pavia_c = PaviaC()
    #ksc = KennedySpaceCenter()
    #botswana = Botswana()

