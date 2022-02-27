import yaml
from matplotlib import pyplot as plt

from psana_interface import *

class TimetoolInterface:

    def __init__(self, path_config):

        with open(path_config, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.init_calib_data()

    def init_calib_data(self):
        """
        Initialize calib data

        Returns
        -------

        """
        self.edge_pos = []
        self.amp = []
        self.laser_time = []

    def retrieve_calib_data(self):
        """
        Retrieve the pre-calculated edge position and laser timing from data source.

        Returns
        -------

        """
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

    def plot_calib_data(self, png_output=None):
        """
        Plot

        Returns
        -------

        """
        fig = plt.figure(figsize=(4,4), dpi=80)
        plt.plot(self.edge_pos, self.laser_time,
                 'o', color='black',label='edge position')
        plt.xlabel('pixel edge')
        plt.ylabel('laser delay')
        plt.legend()
        if png_output is None:
            plt.show()
        else:
            plt.savefig(png_output)