import numpy as np
from types import SimpleNamespace
import psana
from psana import *
try:
    from PSCalib.GeometryAccess import GeometryAccess
    from psana import EventId
    from psana import setOption
except ImportError:
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess

class PsanaInterface:

    def __init__(self, exp, run, det_type,
                 event_receiver=None, event_code=None, event_logic=True,
                 ffb_mode=False, track_timestamps=False, calibdir=None):
        self.exp = exp # experiment name, str
        self.run = run # run number, int
        self.det_type = det_type # detector name, str
        self.track_timestamps = track_timestamps # bool, keep event info
        self.seconds, self.nanoseconds, self.fiducials = [], [], []
        self.event_receiver = event_receiver # 'evr0' or 'evr1', str
        self.event_code = event_code # event code, int
        self.event_logic = event_logic # bool, if True, retain events with event_code; if False, keep all other events
        self.set_up(det_type, ffb_mode, calibdir)
        self.counter = 0

    def set_up(self, det_type, ffb_mode, calibdir=None):
        """
        Instantiate DataSource and Detector objects; use the run 
        functionality to retrieve all psana.EventTimes.
        
        Parameters
        ----------
        det_type : str
            detector type, e.g. epix10k2M or jungfrau4M
        ffb_mode : bool
            if True, set up in an FFB-compatible style
        calibdir: str
            directory to alternative calibration files
        """
        ds_args=f'exp={self.exp}:run={self.run}:idx'
        if ffb_mode:
            ds_args += f':dir=/cds/data/drpsrcf/{self.exp[:3]}/{self.exp}/xtc'
        
        self.ds = psana.DataSource(ds_args)   
        self.det = psana.Detector(det_type, self.ds.env())
        if self.event_receiver is not None:
            self.evr_det = psana.Detector(self.event_receiver)
        self.runner = next(self.ds.runs())
        self.times = self.runner.times()
        self.max_events = len(self.times)
        if calibdir is not None:
            setOption('psana.calib_dir', calibdir)
        self._calib_data_available()

    def _calib_data_available(self):
        """
        Check whether calibration data is available.
        """
        self.calibrate = True
        evt = self.runner.event(self.times[0])
        if (self.det.pedestals(evt) is None) or (self.det.gain(evt) is None):
            print("Warning: calibration data unavailable, returning uncalibrated data")
            self.calibrate = False
            
    def turn_calibration_off(self):
        """
        Do not apply calibration to images.
        """
        self.calibrate = False
        
    def get_pixel_size(self):
        """
        Retrieve the detector's pixel size in millimeters.
        
        Returns
        -------
        pixel_size : float
            detector pixel size in mm
        """
        if self.det_type == 'Rayonix':
            env = self.ds.env()
            cfg = env.configStore()
            pixel_size_um = cfg.get(psana.Rayonix.ConfigV2).pixelWidth()
        else:
            pixel_size_um = self.det.pixel_size(self.ds.env())
        return pixel_size_um / 1.0e3
    
    def get_wavelength(self):
        """
        Retrieve the detector's wavelength in Angstrom.
        
        Returns
        -------
        wavelength : float
            wavelength in Angstrom
        """
        return self.ds.env().epicsStore().value('SIOC:SYS0:ML00:AO192') * 10.
    
    def get_wavelength_evt(self, evt):
        """
        Retrieve the detector's wavelength for a specfic event.

        Parameters
        ----------
        evt : psana.Event object
            individual psana event
        
        Returns
        -------
        wavelength : float
            wavelength in Angstrom
        """
        ebeam = psana.Detector('EBeam')
        photon_energy = ebeam.get(evt).ebeamPhotonEnergy()
        if np.isinf(photon_energy):
            return self.get_wavelength()
        else:
            lambda_m =  1.23984197386209e-06 / photon_energy # convert to meters using e=hc/lambda
            return lambda_m * 1e10

    def estimate_distance(self):
        """
        Retrieve an estimate of the detector distance in mm.
        
        Returns
        -------
        distance : float
            estimated detector distance
        """
        return -1*np.mean(self.det.coords_z(self.run))/1e3

    def get_camera_length(self, pv=None):
        """
        Retrieve the camera length (clen) in mm.

        Parameters
        ----------
        pv : str
            pv code for distance, optional

        Returns
        -------
        clen : float
            clen, where clen = distance - coffset
        """
        if pv is None:
            if self.det_type == 'jungfrau4M':
                pv = 'CXI:DS1:MMS:06.RBV'
            if self.det_type == 'Rayonix':
                pv = 'MFX:DET:MMS:04.RBV'
            if self.det_type == 'epix10k2M':
                pv = 'MFX:ROB:CONT:POS:Z'
            print(f"PV used to retrieve clen parameter: {pv}")

        return self.ds.env().epicsStore().value(pv)

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

    def skip_event(self, evt):
        """
        We skip an event if:
        - it has a specific event_code and our event_logic is False
        - it does not have that event_code and our event_logic is True.

        Parameters
        ----------
        evt : psana.Event object
            individual psana event

        Returns
        -------
        skip_status : Boolean
            if True, skip event
        """
        skip = False
        if self.event_receiver is not None:
            event_codes = self.evr_det.eventCodes(evt)
            found_event = False
            if self.event_code in event_codes:
                found_event = True
            if ( found_event != self.event_logic ):
                skip = True
        return skip

    def distribute_events(self, rank, total_ranks, max_events=-1):
        """
        For parallel processing. Update self.counter and self.max_events such that
        events will be distributed evenly across total_ranks, and each rank will 
        only process its assigned events. Hack to avoid explicitly using MPI here.
        
        Parameters
        ----------
        rank : int
            current rank
        total_ranks : int
            total number of ranks
        max_events : int, optional
            total number of images desired, option to override self.max_events 
        """
        if max_events == -1:
            max_events = self.max_events
            
        # determine boundary indices between ranks
        split_indices = np.zeros(total_ranks)
        for r in range(total_ranks):
            num_per_rank = max_events // total_ranks
            if r < (max_events % total_ranks):
                num_per_rank += 1
            split_indices[r] = num_per_rank
        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int)   
        
        # update self variables that determine start and end of this rank's batch
        self.counter = split_indices[rank]
        self.max_events = split_indices[rank+1]
        
    def get_images(self, num_images, assemble=True):
        """
        Retrieve a fixed number of images from the run. If the pedestal or gain 
        information is unavailable and unassembled images are requested, return
        uncalibrated images. 
        
        Parameters
        ---------
        num_images : int
            number of images to retrieve (per rank)
        assemble : bool, default=True
            whether to assemble panels into image
            
        Returns
        -------
        images : numpy.ndarray, shape ((num_images,) + det_shape)
            images retrieved sequentially from run, optionally assembled
        """
        # set up storage array
        if assemble:
            images = np.zeros((num_images, 
                               self.det.image_xaxis(self.run).shape[0], 
                               self.det.image_yaxis(self.run).shape[0]))
        else:
            images = np.zeros((num_images,) + self.det.shape())
            
        # retrieve next batch of images
        for counter_batch in range(num_images):
            if self.counter >= self.max_events:
                images = images[:counter_batch]
                print("No more events to retrieve")
                break
                
            else:
                evt = self.runner.event(self.times[self.counter])
                if assemble:
                    if not self.calibrate:
                        raise IOError("Error: calibration data not found for this run.")
                    else:
                        images[counter_batch] = self.det.image(evt=evt)
                else:
                    if self.calibrate:
                        images[counter_batch] = self.det.calib(evt=evt)
                    else:
                        images[counter_batch] = self.det.raw(evt=evt)
                        
                if self.track_timestamps:
                    self.get_timestamp(evt.get(EventId))
                    
                self.counter += 1
             
        return images

class Psana2Interface:
    
    def __init__(self, exp, run, det_type, xtc_dir):
        self.exp = exp
        self.run = run
        self.det_type = det_type
        self._set_up(xtc_dir)
    
    def _set_up(self, xtc_dir):
        """
        Set up objects associated with the datasource and detector.
        The detector class is different in psana2, so its functions
        are here split between a det variable that houses the pixel
        mapping information and a detector variable from which the
        events' images and photon energies can be retrieved.
        Parameters
        ----------
        xtc_dir : str
            path to the xtc2 data
        """
        self.ds = DataSource(exp=self.exp, run=self.run, dir=xtc_dir)
        self.runner = next(self.ds.runs())

        # extract dgram contents
        contents = dict()
        contents['iX'] = self.runner.beginruns[0].scan[0].raw.iX
        contents['iY'] = self.runner.beginruns[0].scan[0].raw.iY
        contents['ipx'] = self.runner.beginruns[0].scan[0].raw.ipx
        contents['ipy'] = self.runner.beginruns[0].scan[0].raw.ipy
        contents['shape'] = tuple(self.runner.beginruns[0].scan[0].raw.det_shape)
        contents['clen'] = self.runner.beginruns[0].scan[0].raw.clen
        
        self.det = SimpleNamespace(**contents)
        self.detector = self.runner.Detector(self.det_type)

#### Miscellaneous functions ####

def retrieve_pixel_index_map(geom):
    """
    Retrieve a pixel index map that specifies the relative arrangement of
    pixels on an LCLS detector.
    
    Parameters
    ----------
    geom : string or GeometryAccess Object
        if str, full path to a psana *-end.data file
        else, a PSCalib.GeometryAccess.GeometryAccess object
    
    Returns
    -------
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)
    """
    if type(geom) == str:
        geom = GeometryAccess(geom)

    temp_index = [np.asarray(t) for t in geom.get_pixel_coord_indexes()]
    pixel_index_map = np.zeros((np.array(temp_index).shape[2:]) + (2,))
    pixel_index_map[:,:,:,0] = temp_index[0][0]
    pixel_index_map[:,:,:,1] = temp_index[1][0]
    
    return pixel_index_map.astype(np.int64)

def assemble_image_stack_batch(image_stack, pixel_index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map.
    Either a batch or a single image can be provided. Modified from skopi.
    
    Parameters
    ----------
    image_stack : numpy.ndarray, 3d or 4d 
        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)
    
    Returns
    -------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    """
    if len(image_stack.shape) == 3:
        image_stack = np.expand_dims(image_stack, 0)
    
    # get boundary
    index_max_x = np.max(pixel_index_map[:, :, :, 0]) + 1
    index_max_y = np.max(pixel_index_map[:, :, :, 1]) + 1
    # get stack number and panel number
    stack_num = image_stack.shape[0]
    panel_num = image_stack.shape[1]

    # set holder
    images = np.zeros((stack_num, index_max_x, index_max_y))

    # loop through the panels
    for l in range(panel_num):
        images[:, pixel_index_map[l, :, :, 0], pixel_index_map[l, :, :, 1]] = image_stack[:, l, :, :]
        
    if images.shape[0] == 1:
        images = images[0]

    return images

def disassemble_image_stack_batch(images, pixel_index_map):
    """
    Diassemble a series of 2D diffraction patterns into their consituent panels. 
    Function modified from skopi.
        
    Parameters
    ----------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)

    Returns
    -------
    image_stack_batch : numpy.ndarray, 3d or 4d 
        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    """
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)

    image_stack_batch = np.zeros((images.shape[0],) + pixel_index_map.shape[:3])
    for panel in range(pixel_index_map.shape[0]):
        idx_map_1 = pixel_index_map[panel, :, :, 0]
        idx_map_2 = pixel_index_map[panel, :, :, 1]
        image_stack_batch[:,panel] = images[:,idx_map_1,idx_map_2]
        
    if image_stack_batch.shape[0] == 1:
        image_stack_batch = image_stack_batch[0]
        
    return image_stack_batch
