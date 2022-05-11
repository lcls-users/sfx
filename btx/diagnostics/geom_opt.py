import numpy as np
import sys
from btx.diagnostics.run import RunDiagnostics
from btx.interfaces.psana_interface import assemble_image_stack_batch
from .ag_behenate import *

class GeomOpt:
    
    def __init__(self, exp, run, det_type):
        self.diagnostics = RunDiagnostics(exp=exp, # experiment name, str
                                          run=run, # run number, int
                                          det_type=det_type) # detector name, str
        
    def opt(self, powder, sample='AgBehenate', mask=None, center=None, plot=None):
        """
        Estimate the sample-detector distance based on the properties of the powder
        diffraction image. Currently only implemented for silver behenate.
        
        Parameters
        ----------
        powder : str or int
            if str, path to the powder diffraction in .npy format
            if int, number of images from which to compute powder 
        sample : str
            sample type, currently implemented for AgBehenate only
        mask : str
            npy file of mask in psana unassembled detector shape
        center : tuple
            detector center (xc,yc) in pixels. if None, assume assembled image center.
        plot : str or None
            output path for figure; if '', plot but don't save; if None, don't plot

        Returns
        -------
        distance : float
            estimated sample-detector distance in mm
        """
        if type(powder) == str:
            powder_img = np.load(powder)
        
        elif type(powder) == int:
            print("Computing powder from scratch")
            self.diagnostics.compute_run_stats(n_images=powder, powder_only=True)
            if self.diagnostics.psi.det_type != 'Rayonix':
                powder_img = assemble_image_stack_batch(self.diagnostics.powders['max'], 
                                                        self.diagnostics.pixel_index_map)
        
        else:
            sys.exit("Unrecognized powder type, expected a path or number")
        
        if mask:
            print(f"Applying mask {mask} to powder")
            mask = np.load(mask)
            if self.diagnostics.psi.det_type != 'Rayonix':
                mask = assemble_image_stack_batch(mask, self.diagnostics.pixel_index_map)

        if sample == 'AgBehenate':
            ag_behenate = AgBehenate(powder_img, 
                                     mask       = mask,
                                     pixel_size = self.diagnostics.psi.get_pixel_size(),
                                     wavelength = self.diagnostics.psi.get_wavelength(),)
            ag_behenate.opt_geom(distance_i   = self.diagnostics.psi.estimate_distance(),
                                 n_iterations = 3,
                                 center_i     = center,
                                 plot         = plot)
            return None

        else:
            print("Sorry, currently only implemented for silver behenate")
            return -1
