import numpy as np
import psana
from psana import EventId

class PsanaInterface:

    def __init__(self, exp, run,
                 det_type=None,
                 track_timestamps=False,
                 parallel=False,
                 small_data=False):
        self.exp = exp # experiment name, string
        self.run = run # run number, int
        self.det_type = det_type # detector name, string
        self.track_timestamps = track_timestamps # bool, keep event info
        self.seconds, self.nanoseconds, self.fiducials = [], [], []

        psana_keyword = f'exp={exp}:run={run}'
        if small_data:
            psana_keyword = f'{psana_keyword}:smd'

        if parallel:
            self.ds = psana.MPIDataSource(psana_keyword)
        else:
            self.ds = psana.DataSource(psana_keyword)

        self.det = None
        if det_type is not None:
            self.det = psana.Detector(det_type, self.ds.env())
        
    def get_pixel_size(self):
        """
        Retrieve the detector's pixel size in millimeters.
        
        Returns
        -------
        pixel_size : float
            detector pixel size in mm
        """
        return self.det.pixel_size(self.ds.env()) / 1.0e3
    
    def get_wavelength(self):
        """
        Retrieve the detector's wavelength in Angstrom.
        
        Returns
        -------
        wavelength : float
            wavelength in Angstrom
        """
        return self.ds.env().epicsStore().value('SIOC:SYS0:ML00:AO192') * 10.
    
    def estimate_distance(self):
        """
        Retrieve an estimate of the detector distance in mm.
        
        Returns
        -------
        distance : float
            estimated detector distance
        """
        return -1*np.mean(self.det.coords_z(self.run))/1e3

    def get_timestamp(self, evtId):
        """
        Retrieve the timestamp (seconds, nanoseconds, fiducials) associated with the input 
        event and store in self variables. For further details, see the example here:
        https://confluence.slac.stanford.edu/display/PSDM/Jump+Quickly+to+Events+Using+Timestamps
        
        Parameters
        ----------
        evtId : psana.EventId
            the event ID associated with a particular image
        """
        self.seconds.append(evtId.time()[0])
        self.nanoseconds.append(evtId.time()[1])
        self.fiducials.append(evtId.fiducials())
        return
    
    def get_images(self, num_images, assemble=True):
        """
        Retrieve a fixed number of images from the run. If the pedestal or gain 
        information is unavailable and unassembled images are requested, return
        uncalibrated images.
        
        Paramters
        ---------
        num_images : int
            number of images to retrieve
        assemble : bool, default=True
            whether to assemble panels into image
            
        Returns
        -------
        images : numpy.ndarray, shape ((num_images,) + det_shape)
            images retrieved sequentially from run, optionally assembled
        """
        counter = 0
        calibrate = True

        if assemble:
            images = np.zeros((num_images, 
                               self.det.image_xaxis(self.run).shape[0], 
                               self.det.image_yaxis(self.run).shape[0]))
        else:
            images = np.zeros((num_images,) + self.det.shape())
        
        for num,evt in enumerate(self.ds.events()):
            if counter < num_images:
                # check that pedestal and gain information are available
                if counter == 0:
                    if (self.det.pedestals(evt) is None) or (self.det.gain(evt) is None):
                        calibrate = False
                        if not calibrate and not assemble:
                            print("Warning: calibration data unavailable, returning uncalibrated data")

                # retrieve image, by default calibrated and assembled into detector format
                if assemble:
                    if not calibrate:
                        raise IOError("Error: calibration data not found for this run.")
                    else:
                        images[counter] = self.det.image(evt=evt)
                else:
                    if calibrate:
                        images[counter] = self.det.calib(evt=evt)
                    else:
                        images[counter] = self.det.raw(evt=evt)

                # optionally store timestamps associated with images
                if self.track_timestamps:
                    self.get_timestamp(evt.get(EventId))

            else:
                break
            counter += 1

        if counter < num_images:
            print("It appears we have reached the end of the run")
            images = images[:counter]
            
        return images
