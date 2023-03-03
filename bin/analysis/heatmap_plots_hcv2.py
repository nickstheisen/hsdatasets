#!/usr/bin/env python

from hsdatasets.groundbased import GroundBasedHSDataset
from hsdatasets.analysis.tools import SpectrumPlotter
from hsdatasets.groundbased.prep import download_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from hsdatasets.transforms import ToTensor, PermuteData, ReplaceLabels

transform = transforms.Compose([
            ToTensor(half_precision=True),
            PermuteData(new_order=[2,0,1]),
            ReplaceLabels({255:19})
        ])


n_classes = 19
class_def = '/home/hyperseg/git/hsdatasets/labeldefs/HCv2_labels.txt'
filepath = '/home/hyperseg/data/HyperspectralCityV2.h5'
dataset = GroundBasedHSDataset(filepath, transform=transform)

splotter = SpectrumPlotter(dataset=dataset,
                            dataset_name='hcv',
                            num_classes=n_classes,
                            class_def=class_def)
splotter.extract_class_samples()

splotter.plot_heatmap(out_dir="/mnt/data/hcv_heatmaps/",
                        filetype='jpg')
