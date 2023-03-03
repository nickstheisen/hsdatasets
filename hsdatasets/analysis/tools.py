#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from hsdatasets.utils import load_label_def
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tr



class SpectrumPlotter():
    def __init__(
            self,
            dataset: Dataset,
            dataset_name: str,
            num_classes: int,
            class_def: str  ):
        self.dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                persistent_workers=True)
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.label_names, self.label_colors = load_label_def(class_def)
        self.label_colors = self.label_colors / 255.
        self.class_spectra = None
        self.y_max = None
        self.y_min = None

    def _prepare_for_plotting(self, data):
        ## To efficiently plot the data we need to create one huge list containing all spectra 
        ## one after another. The spectra are seperated from each other by nan-values. Further, 
        ## DataShader uses pandas DataFrames so we need to convert the data to such.
        df = pd.DataFrame(data.T)
        
        ## append row with nan-values, these will be our seperators later
        df = pd.concat([df, 
            pd.DataFrame(
            [np.array([np.nan]*len(df.columns))], columns=df.columns, index=[np.nan]
            )]
        )
        x, y = df.shape

        ## rearrange 2D-array to get a 1D-Array where each column from the original array 
        ## is written after another.
        arr = df.values.reshape((x * y, 1), order='F')

        ## convert this list back to DataFrame as a column with header 'y' and use the x-values as
        ## index values
        df_r = pd.DataFrame(arr, columns=list('y'), index=np.tile(df.index.values, y)) 
        ## by resetting index, the current index value are converted to a new column (0) which
        ## will be used as x-values for plotting
        df_r = df_r.reset_index()
        df_r.columns.values[0] = 'x'
        return df_r

    def extract_class_samples(self):
        if self.class_spectra is not None:
            print("Already extracted class samples. Returning...")
            return
        
        # temporary dictionary to store class samples of each image in
        spectra_dict = dict()
        for c in range(self.num_classes):
            spectra_dict[c] = []
        print(spectra_dict)
        
        # iterate over dataset and extract class spectra
        print("#### Start Extraction ####")
        for data, labels in tqdm(self.dataloader):

            # calculate y-limits
            if self.y_max is None or data.max() > self.y_max:
                self.y_max = data.max()
            if self.y_min is None or data.min() < self.y_min:
                self.y_min = data.min()
            
            # convert to correct shape
            n_channels = data.shape[1]
            data = np.squeeze(data).reshape(n_channels, -1).swapaxes(0,1)
            labels = np.squeeze(labels).reshape(-1)
            
            for c in range(self.num_classes):
                class_spectra = data[labels == c].numpy()

                # prepare data for efficient plotting
                dataframe = self._prepare_for_plotting(class_spectra)
                spectra_dict[c].append(dataframe)
        
        print("#### Extraction finished ####")
        self.y_min = float(self.y_min)
        self.y_max = float(self.y_max)
        print("#### Aggregating Data ####")
        # store class spectra for further processing and analysis
        self.class_spectra = dict()
        for c in range(self.num_classes):
            if len(spectra_dict[c]) == 0:
                self.class_spectra[c] = None
                print(f"Warning: No samples of class '{self.label_names[c]}' with ID '{c}' found.")
                continue
            self.class_spectra[c] = pd.concat(spectra_dict[c])
        print("#### Aggregation Finished ####")
            
    def plot_color(self, out_dir, filetype='jpg',ylim=[0.0,2.0]):
        out_path = Path(out_dir)
        
        # iterate over dataset
        for data, labels in tqdm(self.dataloader):
            n_channels = data.shape[1]
            data = np.squeeze(data).reshape(n_channels, -1).swapaxes(0,1)
            labels = np.squeeze(labels).reshape(-1)
            
            # plot spectra
            for c in range(self.num_classes):
                plt.figure(c)
                class_spectra = data[labels == c]
                plt.plot(class_spectra.T, color=self.label_colors[c], alpha=0.02, linewidth=0.4)
        
        # export plots
        for c in range(self.num_classes):
            plt.figure(c)
            plt.title(self.label_names[c])
            plt.gca().set_ylim(ylim)
            plt.savefig(out_path.joinpath(f"{self.dataset_name}_class{c:02d}.{filetype}"))

    """
    based on: https://stackoverflow.com/questions/47175398/line-based-heatmap-or-2d-line-histogram
    """
    def plot_heatmap(self, out_dir, filetype='jpg'): 
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.class_spectra is None:
            print("Error: Spectra must be extracted before plotting. Please call "
                   " function `extract_class_samples()`.")

        data, labels = next(iter(self.dataloader))
        for c in range(self.num_classes):
            filename = out_path.joinpath(f"{self.dataset_name}_class{c:02d}.{filetype}")
            df = self.class_spectra[c]
            if df is None:
                continue

            # plotting params
            x_range = (df['x'].min(), df['x'].max())
            y_range = (self.y_min, self.y_max)
            ## binning granularity
            w = 1500
            h = 1000
            dpi = 150
            cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=h, plot_width=w)
            
            # aggregate data
            ## use column x as x-values and y for y-vales in plotting then count how many times a
            ## line passed one position in the plot
            aggs = cvs.line(df, 'x', 'y', ds.count())

            ## plot data with one color
            stacked_img = tr.Image(tr.shade(aggs, cmap=plt.cm.Spectral_r))

            # export data
            fig = plt.figure(c)
            ax = fig.add_subplot(111)
            ax.imshow(stacked_img.to_pil())
            plt.title(self.label_names[c])
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.clf()
