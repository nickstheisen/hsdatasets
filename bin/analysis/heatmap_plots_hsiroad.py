#!/usr/bin/env python

from hsdatasets.groundbased import HSIRoadDataset
from hsdatasets.analysis.tools import SpectrumPlotter
from hsdatasets.groundbased.prep import download_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from hsdatasets.transforms import ToTensor, PermuteData, ReplaceLabels

transform = transforms.Compose([
    ToTensor(),
])


n_classes = 2
class_def = '/home/hyperseg/git/hsdatasets/labeldefs/hsi_road_label_def.txt'
filepath = '/home/hyperseg/data/hsi_road/hsi_road'
dataset = HSIRoadDataset(
            data_dir=filepath, 
            transform=transform,
            mode='full',
            collection='nir')

splotter = SpectrumPlotter(dataset=dataset,
                            dataset_name='hsiroad',
                            num_classes=n_classes,
                            class_def=class_def)
splotter.extract_class_samples()

splotter.plot_heatmap(out_dir="/mnt/data/hsiroad_heatmaps/",
                        filetype='jpg')
