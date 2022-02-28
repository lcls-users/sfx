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
                self.cali_edge_pos = data[0]
                self.cali_amp = data[1]
                self.cali_laser_time = data[2]
                return

        psi = PsanaInterface(exp=self.config.exp,
                             run=self.config.calib_run,
                             parallel=self.config.parallel,
                             small_data=True)

        epics_store = psi.ds.env().epicsStore()

        for idx, evt in enumerate(psi.ds.events()):
            self.cali_edge_pos = np.append(self.edge_pos,
                                      epics_store.value(self.config.pv_fltpos))
            self.cali_amp = np.append(self.amp,
                                 epics_store.value(self.config.pv_amp))
            self.cali_laser_time = np.append(self.laser_time,
                                        epics_store.value(self.config.pv_lasertime))

        if save:
            print(f'saving calib data to {calib_data_file}')
            np.save(calib_data_file,
                    np.row_stack((self.cali_edge_pos,
                                     self.cali_amp,
                                     self.cali_laser_time)))

    def plot_calib_data(self, png_output=None):
        """
        Plot

        Returns
        -------

        """
        if hasattr(self, 'calib_model'):
            model_time = self.calib_model[0] + self.calib_model[1]*self.cali_edge_pos + self.calib_model[2]*self.cali_edge_pos**2
        
        fig = plt.figure(figsize=(4,4), dpi=80)
        plt.plot(self.cali_edge_pos, self.cali_laser_time,
                 'o', color='black',label='edge position')
        if hasattr(self, 'calib_model'):
            plt.plot(self.cali_edge_pos, model_time, color='red',label = 'calibration fit')
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
        
        self.calib_model = np.flip(np.polyfit(self.cali_edge_pos.astype('float'), self.cali_laser_time.astype('float'), int(poly)))
        if poly == 1:
            self.calib_model = np.append(self.calib_model, 0)
            
    def filter_calib_data(self, edge_bound, time_bound):
        
        """
        Filters the data used for calibration
        
        """
        self.calib_mask = np.where((self.cali_edge_pos > edge_bound[0]) & (self.cali_edge_pos <edge_bound[1]) & (self.cali_laser_time > time_bound[0]) & (self.cali_laser_time < time_bound[1]), True, False)
        self.cali_edge_pos = self.cali_edge_pos[self.calib_mask]
        self.cali_laser_time = self.cali_laser_time[self.calib_mask]
        
    def get_TTdiagnostics(self, run, save=True):
        """
        Retrieve TT output from psana
        """
        
        
        runs = np.arange(self.run_start, self.run_end+1)
            
        TTdata_file = f'{self.config.work_dir}TT_data_r{run}.npy'

        psana_keyword=f'exp={self.exp}:run={run}'
        print(psana_keyword)
        if self.parallel:
           ds = MPIDataSource(f'{psana_keyword}:smd')
        else:
           ds = psana.DataSource(f'{psana_keyword}:smd') 

        epics_store = ds.env().epicsStore()
        for idx, evt in enumerate(psi.ds.events()):
            self.edge_pos = np.append(self.edge_pos,
                                      epics_store.value(self.config.pv_fltpos))
            self.amp = np.append(self.amp,
                                 epics_store.value(self.config.pv_amp))
            self.laser_time = np.append(self.laser_time,
                                        epics_store.value(self.config.pv_lasertime))
            self.TTtime = np.append(self.TTtime,
                                    epics_store.value(self.config.pv_TTtime))

        if save:
            print(f'saving calib data to {TTdata_file}')
            np.save(TTcalib_data_file,
                    np.row_stack((self.edge_pos,
                                  self.amp,
                                  self.laser_time,
                                  self.TTtime)))
                
                
                
    def get_delay(self):
        """
        Retrieve TT delay
        
        write out
        """
        
        
        runs = np.arange(self.run_start, self.run_end+1)
        for run in runs:
            TTdata_file = f'{self.config.work_dir}TT_data_r{run}.npy'
            if os.path.isfile(TTdata_file):
                print(f'found {TTdata_file}')
                data = np.load(TTdata_file, allow_pickle=True)
                self.edge_pos = data[0]
                self.amp = data[1]
                self.laser_time = data[2]
            else:
                self.get_TTdiagnostics(run)
            
            if self.redoTT:
                #here the TT analysis should be redone
                continue
                
            elif hasattr(self, 'calib_model'):
                #TT correction is calculated from psana output and calibration
                self.get_TTtime()
                
            else:
                #read TT correction directly from psana
                self.TTtime = data[3]
     
    
    def get_TTtime(self):
        """
        Translate edge position to delay
        """
        self.TTtime = self.calib_model[0] + self.edge_pos*self.calib_model[1] + self.edge_pos**2 *self.calib_model[2] 

        
        
    def filter_TTdata(self):
        """
        Filter data according to FWHM and amplitude
        """
        

        
