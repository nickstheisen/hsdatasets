# hsdatasets

The `hsdatasets`-package provides pytorch-DataSet wrappers for the most common hyperspectral
data sets with pixel-precise ground-truth annotations. This simplifies the usage of those data sets
in deep learning applications. 

Currently only wrapper classes for remote sensing data sets are provided but in the future other
data sets such as [HyKo2](https://wp.uni-koblenz.de/hyko/dataset/) will be provided as well.

## Remote Sensing Data

### Data Sampling

After loading the data the image is zero-padded and NxN patches are sampled at each pixel position.
If necessary the dimensionality can be reduced to M dimensions using PCA. The user can define M and 
N during class instantiation.

**Warning: If N>1 pixels in multiple data patches overlap which may lead to data leakage when not taken care of.**

### Currently Supported Data Sets

Data Set|Spatial Resolution [px]|Spectral Resolution [bands] | Classes |Sensor
---|---|---|---|---
[Indian Pines](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines)|145 x 145| 200 | 16 |[AVIRIS](https://aviris.jpl.nasa.gov/)
[Salinas Scene](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene)|512 x 217| 204 | 16 | [AVIRIS](https://aviris.jpl.nasa.gov/)
[Salinas-A](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas-A_scene)|86 x 83 | 204 | 6 | [AVIRIS](https://aviris.jpl.nasa.gov/)
[Kennedy Space Center](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Kennedy_Space_Center_.28KSC.29)|512 x 614|176|13| [AVIRIS](https://aviris.jpl.nasa.gov/)
[Pavia Centre](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene)|1096 x 1096 | 102 | 9 | [ROSIS](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/4545/0000/HySens-DAISROSIS-Imaging-Spectrometers-at-DLR/10.1117/12.453677.short)
[Pavia University](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene)|619 x 610 | 103 | 9 | [ROSIS](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/4545/0000/HySens-DAISROSIS-Imaging-Spectrometers-at-DLR/10.1117/12.453677.short)
[Botswana](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Botswana)|1476 x 256 | 145 | 14 | [Hyperion](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-earth-observing-one-eo-1-hyperion?qt-science_center_objects=0#qt-science_center_objects)
[AeroRIT](https://github.com/aneesh3108/AeroRIT)|1973 x 3975|51|5|[Headwall Hyper E](https://www.headwallphotonics.com/hyperspectral-sensors)
