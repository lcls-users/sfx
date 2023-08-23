import os, csv, argparse

import numpy as np
from mpi4py import MPI

from matplotlib import pyplot as plt
from matplotlib import colors

import holoviews as hv
hv.extension('bokeh')
from holoviews.streams import Params

import panel as pn
import panel.widgets as pnw

from btx.misc.shortcuts import TaskTimer

from btx.interfaces.ipsana import (
    PsanaInterface,
    bin_data,
    bin_pixel_index_map,
    retrieve_pixel_index_map,
    assemble_image_stack_batch,
)

class DimRed:

    """Dimension Reduction Parent Class."""

    def __init__(
        self,
        exp,
        run,
        det_type,
        start_offset=0,
        num_images=10,
        num_components=10,
        batch_size=10,
        priming=False,
        downsample=False,
        bin_factor=2,
        output_dir=""
    ):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
        self.psi.counter = start_offset
        self.start_offset = start_offset

        self.priming = priming
        self.downsample = downsample
        self.bin_factor = bin_factor
        self.output_dir = output_dir

        (
            self.num_images,
            self.num_components,
            self.batch_size,
            self.num_features,
        ) = self.set_params(num_images, num_components, batch_size, bin_factor)

        self.split_indices, self.split_counts = distribute_indices_over_ranks(
            self.num_features, self.size
        )

        self.task_durations = dict({})

        self.num_incorporated_images = 0
        self.outliers, self.pc_data = [], []

    def get_params(self):
        """
        Method to retrieve dimension reduction parameters.

        Returns
        -------
        num_incorporated_images : int
            number of images used to build model
        num_components : int
            number of components maintained in model
        batch_size : int
            batch size used in model updates
        num_features : int
            dimensionality of incorporated images
        """
        return (
            self.num_incorporated_images,
            self.num_components,
            self.batch_size,
            self.num_features,
        )

    def set_params(self, num_images, num_components, batch_size, bin_factor):
        """
        Method to initialize dimension reduction parameters.

        Parameters
        ----------
        num_images : int
            Desired number of images to incorporate into model.
        num_components : int
            Desired number of components for model to maintain.
        batch_size : int
            Desired size of image block to be incorporated into model at each update.
        bin_factor : int
            Factor to bin data by.

        Returns
        -------
        num_images : int
            Number of images to incorporate into model.
        num_components : int
            Number of components for model to maintain.
        batch_size : int
            Size of image block to be incorporated into model at each update.
        num_features : int
            Number of features (dimension) in each image.
        """
        max_events = self.psi.max_events
        downsample = self.downsample

        num_images = min(num_images, max_events) if num_images != -1 else max_events
        num_components = min(num_components, num_images)
        batch_size = min(batch_size, num_images)

        # set d
        det_shape = self.psi.det.shape()
        num_features = np.prod(det_shape).astype(int)

        if downsample:
            if det_shape[-1] % bin_factor or det_shape[-2] % bin_factor:
                print("Invalid bin factor, toggled off downsampling.")
                self.downsample = False
            else:
                num_features = int(num_features / bin_factor**2)

        return num_images, num_components, batch_size, num_features

    def display_dashboard(self):
        """
        Displays a pipca dashboard with a PC plot and intensity heatmap.
        """

        start_img = self.start_offset

        # Create PC dictionary and widgets
        PCs = {f'PC{i}' : v for i, v in enumerate(self.pc_data, start=1)}
        PC_options = list(PCs)

        PCx = pnw.Select(name='X-Axis', value='PC1', options=PC_options)
        PCy = pnw.Select(name='Y-Axis', value='PC2', options=PC_options)
        widgets_scatter = pn.WidgetBox(PCx, PCy, width=100)

        tap_source = None
        posxy = hv.streams.Tap(source=tap_source, x=0, y=0)

        # Create PC scatter plot
        @pn.depends(PCx.param.value, PCy.param.value)
        def create_scatter(PCx, PCy):
            img_index_arr = np.arange(start_img, start_img + len(PCs[PCx]))
            scatter_data = {**PCs, 'Image': img_index_arr}

            opts = dict(width=400, height=300, color='Image', cmap='rainbow',
                        colorbar=True, show_grid=True, toolbar='above', tools=['hover'])
            scatter = hv.Points(scatter_data, kdims=[PCx, PCy], vdims=['Image'],
                                label="%s vs %s" % (PCx.title(), PCy.title())).opts(**opts)

            posxy.source = scatter
            return scatter

        # Define function to compute heatmap based on tap location
        def tap_heatmap(x, y, pcx, pcy):
            # Finds the index of image closest to the tap location
            img_source = None
            min_diff = None
            square_diff = None

            for i, (xv, yv) in enumerate(zip(PCs[pcx], PCs[pcy])):
                square_diff = (x - xv) ** 2 + (y - yv) ** 2
                if (min_diff is None or square_diff < min_diff):
                    min_diff = square_diff
                    img_source = i

            # Downsample so heatmap is at most 100 x 100
            counter = self.psi.counter
            self.psi.counter = start_img + img_source
            img = self.psi.get_images(1)
            _, x_pixels, y_pixels = img.shape
            self.psi.counter = counter

            max_pixels = 100
            bin_factor_x = int(x_pixels / max_pixels)
            bin_factor_y = int(y_pixels / max_pixels)

            while x_pixels % bin_factor_x != 0:
                bin_factor_x += 1
            while y_pixels % bin_factor_y != 0:
                bin_factor_y += 1

            img = img.reshape((x_pixels, y_pixels))
            binned_img = img.reshape(int(x_pixels / bin_factor_x),
                                        bin_factor_x,
                                        int(y_pixels / bin_factor_y),
                                        bin_factor_y).mean(-1).mean(1)

            # Creates hm_data array for heatmap
            bin_x_pixels, bin_y_pixels = binned_img.shape
            rows = np.tile(np.arange(bin_x_pixels).reshape((bin_x_pixels, 1)), bin_y_pixels).flatten()
            cols = np.tile(np.arange(bin_y_pixels), bin_x_pixels)

            hm_data = np.stack((rows, cols, binned_img.flatten()))
            hm_data = hm_data.T.reshape((bin_x_pixels * bin_y_pixels, 3))

            opts = dict(width=400, height=300, cmap='plasma', colorbar=True, toolbar='above')
            heatmap = hv.HeatMap(hm_data, label="Image %s" % (start_img+img_source)).aggregate(function=np.mean).opts(**opts)

            return heatmap

        # Connect the Tap stream to the tap_heatmap callback
        stream1 = [posxy]
        stream2 = Params.from_params({'pcx': PCx.param.value, 'pcy': PCy.param.value})
        tap_dmap = hv.DynamicMap(tap_heatmap, streams=stream1+stream2)

        return pn.Row(widgets_scatter, create_scatter, tap_dmap).servable('Cross-selector')


def distribute_indices_over_ranks(d, size):
    """

    Parameters
    ----------
    d : int
        total number of dimensions
    size : int
        number of ranks in world

    Returns
    -------
    split_indices : ndarray, shape (size+1 x 1)
        division indices between ranks
    split_counts : ndarray, shape (size x 1)
        number of dimensions allocated per rank
    """

    total_indices = 0
    split_indices, split_counts = [0], []

    for r in range(size):
        num_per_rank = d // size
        if r < (d % size):
            num_per_rank += 1

        split_counts.append(num_per_rank)

        total_indices += num_per_rank
        split_indices.append(total_indices)

    split_indices = np.array(split_indices)
    split_counts = np.array(split_counts)

    return split_indices, split_counts

