# hsdatasets

The 'hsdatasets'-package provides pytorch-DataSet wrappers for the most common hyperspectral
data sets with pixel-precise ground-truth annotations. This simplifies the usage of those data sets
in deep learning applications. 

Currently only wrapper classes for remote sensing data sets are provided but in the future other
data sets such as [HyKo2](https://wp.uni-koblenz.de/hyko/dataset/) will be provided as well.

## remote sensing data

### data sampling

After loading the data the image is zero-padded and NxN patches are sampled at each pixel position.
If necessary the dimensionality can be reduced to M dimensions using PCA. The user can define M and 
N during class instantiation.
