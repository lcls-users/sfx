"""!
@brief Utility classes and functions for accessing YAML files, checking for 
file existence and fetching the latest files.

Classes:
AttrDict

Functions:
fetch_latest
check_file_existence
"""

import numpy as np
import os
import glob
import time

class AttrDict(dict):
    """! Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)

    Adapted from: https://stackoverflow.com/a/48806603
    """

    def __init__(self, mapping=None, *args, **kwargs):
        """! Return a class with attributes equal to the input dictionary.
        Parameters
        ----------
        @param mapping (bool)
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

def fetch_latest(fnames, run):
    """! Fetch the most recently created (in terms of run numbers) file.
    Here we assume that files are named /{base_path}/r{run:04}.* .
    
    Parameters
    ----------
    @param fnames (str) glob-expandable string pointing to geom or mask files
    @param run (int) run number
    
    Returns
    -------
    @return fname (str) filename of relevant geom or mask file
    """
    fnames = glob.glob(fnames)
    avail = [os.path.basename(f)[1:].split('.')[0] for f in fnames]
    avail = np.array([int(a) for a in avail])
    sort_idx = np.argsort(avail)
    idx = np.searchsorted(avail[sort_idx], run, side='right') - 1
    try:
        return fnames[sort_idx[idx]]
    except IndexError:
        print('File not found.')
        return ''

def check_file_existence(fname, timeout, frequency=15):
    """! Pause until a given file exists, exiting if the waiting
    period exceeds timeout.
    
    Parameters
    ----------
    @param fname (str) name of file whose existence to check for
    @param timeout (float) permitted waiting period in seconds
    @param frequency (float) frequency with which to check in seconds
    """
    start_time = time.time() 
    while time.time() - start_time < timeout:
        if os.path.exists(fname):
            break
        time.sleep(frequency)
