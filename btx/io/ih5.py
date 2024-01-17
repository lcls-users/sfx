"""!
@brief Utilities for working with HDF5 files and managing concurrent write operations.

Classes:
SmallDataReader : Interface to HDF5 files written by Small Data.

Functions:
lock_file(path, timeout) : Decorator to prevent concurrent write operations.
figure_to_array(figure) : Convert a plt.Figure to np.ndarray for storage in h5.
array_to_figure(figure) : Convert a np.ndarray back into a plt.Figure.
"""

import os
from types import prepare_class
import numpy as np
import signal
import time
import h5py
import tables
import socket
import matplotlib.pyplot as plt
from typing import Union, Optional, Callable, Any
from .h5terminalapp import *

def lock_file(path:str, timeout: int = 5, wait: float = 0.5) -> Callable:
    """! Decorator mainly for write functions. Will not allow the decorated
    function to execute if an associated '.lock' file exists. If no lock file
    exists one will be created, preventing other decorated functions from
    executing while the current function completes its operations. Attempts at
    function execution will also be abandoned after a timeout threshold is
    reached.

    Usage e.g.:
    @lock_file('shared.h5', timeout=2)
    def my_write_function():
        print('Writing file safely')
    my_write_function() # Create shared.lock if it doesn't exist

    @param path (str) Path including filename of the file to be locked.
    @param timeout (int) Timeout time in seconds to abandon execution attempt.
    @param wait (float) Time in seconds to wait between write attempts.
    """
    def decorator_lock(write_func: Callable) -> Callable:
        """! Inner decorator for file locking.

        @param write_func (function) The function to run while file is locked.
        """
        def wrapper(*args, **kwargs):
            lockfile = path.split('.')[0] + '.lock'
            written = False
            sendMsg = True
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            try:
                while not written:
                    if not os.path.exists(lockfile):
                        with open(lockfile, 'w') as lock:
                            print(f'Locking file {path}.')
                        print(f'Writing file {path}.')
                        write_func()
                        os.remove(lockfile)
                        written = True
                        sendMsg = False
                        print('Unlocking file.')
                    else:
                        if sendMsg:
                            print(f'File {path} is locked.')
                            sendMsg = False
                        time.sleep(wait)
            except TimeoutError as err:
                print(err)
            signal.alarm(0)
        return wrapper
    return decorator_lock


def _timeout_handler(signum, frame):
    raise TimeoutError()


class TimeoutError(Exception):
    """! Error raised if timeout is reached."""

    def __init__(self, msg='Timeout reached'):
        """! Exception initializer."""
        super().__init__(msg)


def figure_to_array(self, figure: plt.Figure) -> np.ndarray:
    """! Convert a matplotlib Figure object to an array.

    Converting a figure to an array allows for simpler storage of the
    data in an hdf5 file. A sister method `array_to_figure` performs the
    reverse operation.

    @param figure (plt.Figure) : Figure in plt.Figure form to convert.
    @return figArr (np.ndarray) : Figre in np.ndarray form.
    """
    from matplotlib.backend.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(figure)
    canvas.draw()

    figArr = np.asarray(canvas.buffer_rgba())
    return figArr


def array_to_figure(self, figArr: np.ndarray) -> plt.Figure:
    """! Convert a Numpy array into a matplotlib Figure.

    Converts a previously converted matplotlib Figure from an array back
    into a Figure either to facilitate display for to allow the figure
    to be stored in another format.

    @param figArr (np.ndarray) : Figure in array form to convert.
    @return figure (plt.Figure) : Figure in plt.Figure form.
    """
    figure = plt.Figure()
    ax = figure.add_subplot(111)
    ax.imshow(figArr)
    return figure


class SmallDataReader:
    """! Class interface for reading SmallData hdf5 files.

    This class defines simple utilities for interacting with data stored in
    hdf5 files. Plotting methods are provided for frequently used categories
    of data. Plots can be output as png files or have their data saved to a
    common hdf5 file to simplify house keeping. Methods for storing and
    retrieving figure/plot data from hdf5 files are provided, along with
    general methods for interacting with the hdf5 file system. SmallDataReader
    instances can also be subscripted identically to h5py File objects to
    directly access stored data.
    """

    def __init__(self, expmt: str,
                 run: Union[str, int],
                 savedir: str,
                 h5path: Optional[str] = None):
        """! SmallDataReader initializer.

        @param expmt (str) : Experiment name.
        @param run (str | int) : Run number or range of hyphen-separated runs.
        @param savedir (str) : Path to write any output files to.
        @param h5path (str | None) : Path to an hdf5 file. Will search if None.
        """
        if h5path:
            self._path = h5path
        else:
            if 'drp' in socket.gethostname():
                self._path = (f"/cds/data/drpsrcf/{expmt[:3]}/{expmt}/scratch/"
                              f"hdf5/smalldata/{expmt}_Run{int(run):04d}")
            else:
                self._path = (f"/cds/data/psdm/{expmt[:3]}/{expmt}/"
                              f"hdf5/smalldata/{expmt}_Run{int(run):04d}")

        self.expmt = expmt
        self.run = run
        self.savedir = savedir

        try:
            self.root = tables.File(self._path).root
            self.h5 = h5py.File(self._path)
        except FileNotFoundError:
            print("No small data available for this run.")

    def write_to_node(self, h5file: str, node: str, data: np.ndarray):
        """! Write data to an HDF5 node.

        Check if the node exists before writing. If so, alter the data,
        otherwise create the node with the data. This method exists to provide
        the boilerplate existence check when writing to a node.

        @param h5file (str) : File to write to.
        @param node (str) : Full path of the node to be written to.
        @param data (array-like) : Data to write to the file.
        """
        with h5py.File(f'{h5file}', 'w') as f:
            if node in f:
                f[node].data = data
            else:
                f[node] = data

    def print_all_nodes(self):
        """! Print all groups and datasets in the HDF5 file."""
        self.traverse_nodes(self.h5, level=1)

    def traverse_nodes(self, node, level: int = 1):
        """! Recurses through an HDF5 file hierarchy printing nodes."""
        if type(node) == h5py._hl.dataset.Dataset:
            return
        else:
            for key in node.keys():
                print('\t'*(level - 1) + f'{key}\n')
                self.traverse_nodes(node[key], level=level + 1)

    def timestamped_data_totext(self, node_list: list, filename: str):
        """! Write a text file of unique event identifiers and h5 node values.

        This can be used to write out e.g. specific evr codes per event. Data
        is written in a space separated format, with one event per line. The
        first column contains event identifiers, followed by the values for
        each node in subsequent columns. E.g.:
        unique-identifier node1        node2        ...
                id_event1 event1-node1 event1-node2
                id_event2 event2-node1 event2-node2
                   ...

        Unique identifiers are of the form: `event_time-fiducial`

        @param node_list (list[str]) List of hdf5 nodes to write to text.
        @param filename (str) Filename to write the text file to.
        """
        stamps = self.unique_stamps()
        info_to_write = [stamps]
        for node in node_list:
            if node in self.h5:
                info_to_write.append(self.h5[node])
            else:
                print(f'Ignoring node: {node}. Not in small data.')

        info_to_write = np.array(info_to_write).T
        np.savetext(f'{self.savedir}/{filename}', '%s')

    def unique_stamps(self) -> np.ndarray:
        """! Combine timestamps and fiducials for unique event identifiers."""
        stamps = list(map(lambda x, y: f'{x}-{y}',
                          self.h5['event_time'][:],
                          self.h5['fiducials'][:]))
        return np.array(stamps)

    def h5_explorer(self):
        """! Interactive exploration of the small data hdf5.

        Launch a terminal program for viewing groups and datasets.
        Can ONLY be used when running Python in calculator mode in terminal.
        This method does not behave properly in Jupyter Notebooks.
        """
        try:
            app = H5TerminalApp(self._path)
            app.run()
        except Exception as e:
            print(e)

    def __getitem__(self, key: str) -> Any:
        """! Overload subscripting operator, [], to access hdf5 file.

        @param key (str) : Node to access.
        @return value (Any) : The value living at the specified key in the hdf5.
        """
        try:
            return self.h5[key]
        except KeyError:
            print(f'Incorrect node: {key}.')

    def __setitem__(self, key: str, value: Any):
        """! Provided only to prevent accidental errors."""
        print('Writing to hdf5 prohibited.')

    # Task specific plotting functions
    # - One plotting method per task
    # - Also one plotting method per figure
    # - If a task requires multiple figures, call the multiple figure methods
    #   from the 'task' method. See plot_timetool_diagnostics for example.
    #########################################################################
    def plot_image_sums(self, output_type: str = 'png'):
        """! Plots the sums of the images over a run for the used detectors.

        The image sums (powder patterns for crystallography) can be used to
        verify that data was collected during a run.
        """
        if 'Sums' in self.h5:
            imgList = [self.h5[f'Sums/{img}']
                       for img in self.h5['Sums'].keys()]

            for img in imgList:
                squeezed = img[:].squeeze()
                if len(squeezed.shape) > 2:
                    self.construct_img_panels(img)
                else:
                    lims = (np.nanpercentile(squeezed, 5),
                            np.nanpercentile(squeezed, 95))
                    title = img.name.split('/')[2]
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)
                    ax.imshow(squeezed, clim=lims)
                    ax.set_title(title)
                    fig.tight_layout()
                    fig.savefig(f'{self.savedir}/{title}_Sum_r{self.run:04d}')
        else:
            print('Image sums are not available.')


    def construct_img_panels(self):
        """! Assemble multipanel detector images for plotting."""
        pass

    def plot_timetool_diagnostics(self, output_type: str = 'png'):
        """! Calls all timetool related plotting functions.

        Plots can be written out to png figures or stored in an hdf5 file.

       @param output_type (str) : Output format for plots. Either h5 or png.
        """
        self.plot_timetool_histograms(output_type)
        self.plot_timetool_amplitude_correlation(output_type)

    def plot_timetool_histograms(self, output_type: str = 'png'):
        """! Plot histograms of timetool output.

        Plots the histograms of: the timetool correction, the convolution
        amplitudes, and the full-width half-maximums (FWHMs) of the
        convolutions.

        @param output_type (str) : Output format for plots. Either h5 or png.
        """
        if 'tt' in self.h5:
            fltpos_ps = self.h5['tt/FLTPOS_PS'][:]
            ampl = self.h5['tt/AMPL'][:]
            fwhm = self.h5['tt/FLTPOSFWHM'][:]

            fig, axs = plt.subplots(1, 3, figsize=(6, 4), dpi=200)
            axs[0].hist(fltpos_ps, bins=30, density=True)
            axs[0].set_title('Timetool Correction')
            axs[0].set_xlabel('Correction (ps)')
            axs[0].set_ylabel('Density')
            axs[1].hist(ampl, bins=30, density=True)
            axs[1].set_title('Convolution\nAmplitude')
            axs[1].set_xlabel('Amplitude (a.u.)')
            axs[2].hist(fwhm, bins=30, density=True)
            axs[2].set_title('Convolution FWHM')
            axs[2].set_xlabel('FWHM (a.u.)')
            fig.tight_layout()
            if output_type.lower() == 'png':
                fig.savefig(f'{self.savedir}/ttHistograms_r{self.run:04d}.png')
            elif output_type.lower() == 'h5':
                pass
        else:
            print('No timetool data available.')

    def plot_timetool_amplitude_correlation(self,
                                            xrayinode: Optional[str] = None,
                                            output_type: str = 'h5'):
        """! Plot the 2D histogram of x-ray intensity and convolution amplitude.

        The timetool convolution amplitude should be correlated with the x-ray
        intensity. The 2D histogram of these events provides a visual metric
        that the timetool results are reasonable. The correlations hould appear
        to be ~linear.

        @param xrayinode (str) : Specific node with X-ray intensity recordings.
        @param output_type (str) : Output format for plots. Either h5 or png.
        """
        try:
            ampl = self.h5['tt/AMPL'][:]

            if xrayinode:
                xray_i = self.h5[f'{xrayinode}/sum'][:]
            elif (xrayinodes := self.find_xray_intensity_nodes()):
                xray_i = self.h5[f'{xrayinodes[0]}/sum'][:]
            else:
                xray_i = []

            if len(xray_i) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
                ax.hexbin(xray_i, ampl, mincnt=1)
                ax.set_xlabel('X-ray Intensity')
                ax.set_ylabel('Timetool Convolution Amplitude')
                fig.tight_layout()
                if output_type.lower() =='png':
                    fig.savefig(f'{self.savedir}/ttAmplXrayICorr_r{self.run:04d}.png')
                elif output_type.lower() == 'h5':
                    pass

        except KeyError as err:
            print(f'Node not found: {str(err).split()[6]}')

    def find_xray_intensity_nodes(self) -> list:
        """! Determines the available readouts of X-ray intensity."""
        keys = [key for key in self.h5.keys()
                if 'ipm' in key and 'sum' in self.h5[key].keys()]
        # keys is an empty list if no nodes found
        return keys



