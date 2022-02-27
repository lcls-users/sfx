import os,yaml
from matplotlib import pyplot as plt

from psana_interface import *

class AttrDict(dict):
    """Class to convert a dictionary to a class.

    Parameters
    ----------
    dict: dictionary

    """

    def __init__(self, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TimetoolInterface:

    def __init__(self, path_config):

        with open(path_config, "r") as config_file:
            self.config = AttrDict(yaml.safe_load(config_file))

        self.init_job()
        self.init_calib_data()

    def init_job(self):
        """
        Initialize job
        Returns
        -------

        """
        if not os.path.exists(self.config.work_dir):
            os.makedirs(self.config.work_dir)

    def init_calib_data(self):
        """
        Initialize calib data

        Returns
        -------

        """
        self.edge_pos = []
        self.amp = []
        self.laser_time = []

    def retrieve_calib_data(self, save=False, overwrite=False):
        """
        Retrieve the pre-calculated edge position and laser timing from data source.

        Returns
        -------

        """
        calib_data_file = f'{self.config.work_dir}calib_data_r{self.config.calib_run}.npy'
        if os.path.isfile(calib_data_file):
            print(f'found {calib_data_file}')
            if not overwrite:
                print(f'reading calib data from {calib_data_file}')
                data = np.load(calib_data_file, allow_pickle=True)
                self.edge_pos = data[0]
                self.amp = data[1]
                self.laser_time = data[2]
                return

        psi = PsanaInterface(exp=self.config.exp,
                             run=self.config.calib_run,
                             parallel=self.config.parallel,
                             small_data=True)

        epics_store = psi.ds.env().epicsStore()

        for idx, evt in enumerate(psi.ds.events()):
            self.edge_pos = np.append(self.edge_pos,
                                      epics_store.value(self.config.pv_fltpos))
            self.amp = np.append(self.amp,
                                 epics_store.value(self.config.pv_amp))
            self.laser_time = np.append(self.laser_time,
                                        epics_store.value(self.config.pv_lasertime))

        if save:
            print(f'saving calib data to {calib_data_file}')
            np.save(calib_data_file,
                    np.row_stack((self.edge_pos,
                                     self.amp,
                                     self.laser_time)))

    def plot_calib_data(self, png_output=None):
        """
        Plot

        Returns
        -------

        """
        if self.calib_model:
            model_time = self.calib_model[0]**2*self.edge_pos + self.calib_model[1]*self.edge_pos + self.calib_model[2]
        
        fig = plt.figure(figsize=(4,4), dpi=80)
        plt.plot(self.edge_pos, self.laser_time,
                 'o', color='black',label='edge position')
        if self.calib_model:
            plt.plot(self.edge_pos, self.model_time, color='red',label = 'calibration fit')
        plt.xlabel('pixel edge')
        plt.ylabel('laser delay')
        plt.legend()
        if png_output is None:
            plt.show()
        else:
            plt.savefig(png_output)
            
            
    def fit_calib_data(self, poly=1):
        """
        Fits 1st or 2nd order polynomial to data
        
        Returns calibration constants covnerting px to ns
        """
        
        self.model = np.polyfit(self.edge_pos, self.time, int(poly))
        if poly == 1:
            self.model = np.append(self.model, 0)
        
