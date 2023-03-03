#!/usr/bin/env python

from hsdatasets.groundbased import GroundBasedHSDataset
from hsdatasets.analysis.tools import SpectrumPlotter
from hsdatasets.groundbased.prep import download_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from hsdatasets.transforms import ToTensor, PermuteData, ReplaceLabels

transform = transforms.Compose([
    ToTensor(),
    PermuteData(new_order=[2,0,1]),
    ReplaceLabels({0:10, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}) # replace undefined label 0 with 10 and then shift labels by one
])

n_classes = 10
class_def = '/home/hyperseg/git/hsdatasets/labeldefs/hsidrive-labels.txt'
filepath = download_dataset('~/data', 'HyKo2-VIS_Semantic')
dataset = GroundBasedHSDataset(filepath, transform=transform)

splotter = SpectrumPlotter(dataset=dataset,
                            dataset_name='hyko2vissem',
                            num_classes=n_classes,
                            class_def=class_def)
splotter.extract_class_samples()

splotter.plot_heatmap(out_dir="/mnt/data/hyko_heatmaps/",
                        filetype='jpg')
