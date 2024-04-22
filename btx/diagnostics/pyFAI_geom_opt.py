import numpy as np
import pyFAI
from pyFAI.calibrant import CalibrantFactory, CALIBRANT_FACTORY
from pyFAI.goniometer import SingleGeometry
from pyFAI.geometry import Geometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.gui import jupyter
from btx.interfaces.ipsana import *
from btx.diagnostics.converter import CrystFEL_to_PyFAI


class pyFAI_Geometry_Optimization:
    """
    Class to perform Geometry Optimization using pyFAI

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detector : str
        Detector type
    geom : str
        Geometry file in CrystFEL format
    """

    def __init__(
        self, exp, run, detector, geom, n_images=100, max_rings=5, pts_per_deg=1, Imin=0
    ):
        self.exp = exp
        self.run = run
        self.det_type = detector
        self.detector = self.load_geometry(geom)
        self.powder = self.load_powder_data()
        self.geometry = self.pyFAI_optimization()

    def load_geometry(self, geom):
        """
        Load geometry from CrystFEL format
        Convert to pyFAI format

        Parameters
        ----------
        geom : str
            Geometry file in CrystFEL format
        """
        converter = CrystFEL_to_PyFAI(geom)
        detector = converter.detector
        detector.set_pixel_corners(converter.corner_array)
        return detector

    def load_powder_data(self, n_images=100):
        """
        Load powder data from the specified experiment and run

        Parameters
        ----------
        n_images : int
            Number of images to load
        """
        psi = PsanaInterface(exp=self.exp, run=self.run, det_type=self.det_type)
        print(
            f"Instantiated run {self.run} of exp {self.exp} looking at {self.det_type}"
        )
        psi.calibrate = True
        self.psi = psi
        unassembled_images = psi.get_images(n_images, assemble=False)
        calib_avg = np.mean(unassembled_images, axis=0)
        calib_avg_flat = np.reshape(calib_avg, (16 * 2 * 176, 2 * 192))
        return calib_avg_flat

    def pyFAI_optimization(self, max_rings=5, pts_per_deg=1, Imin=0):
        """
        From guessed initial geometry, optimize the geometry using pyFAI package

        Parameters
        ----------
        max_rings : int
            Maximum number of rings to use for calibration
        pts_per_deg : int
            Number of points per degree to use for calibration (spacing of control points)
        Imin : float
            Minimum intensity to use for calibration
        """
        # 1. Define Calibrant
        behenate = CALIBRANT_FACTORY("AgBh")
        wavelength = self.psi.get_wavelength() * 1e-10
        behenate.wavelength = wavelength

        # 2. Define Guessed Geometry
        dist = self.psi.estimate_distance() * 1e-3
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        poni1 = -np.mean(p1)
        poni2 = -np.mean(p2)
        guessed_geom = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            detector=self.detector,
            wavelength=wavelength,
        )

        # 3. Optimization Loop
        params = [guessed_geom.dist, guessed_geom.poni1, guessed_geom.poni2]
        r = 0
        pixel_size = min(self.detector.pixel1, self.detector.pixel2)
        print(
            f"Starting optimization with initial guess: dist={params[0]:.3f}m, poni1={params[1]/pixel_size:.3f}pix, poni2={params[2]/pixel_size:.3f}pix"
        )
        while True:
            sg = SingleGeometry(
                label=f"step_{r+1}",
                image=self.powder,
                calibrant=behenate,
                detector=self.detector,
                geometry=guessed_geom,
            )
            sg.extract_cp(max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=Imin)
            sg.geometry_refinement.refine3(fix=["rot1", "rot2", "rot3", "wavelength"])
            new_params = [
                sg.geometry_refinement.param[0],
                sg.geometry_refinement.param[1],
                sg.geometry_refinement.param[2],
            ]
            if np.allclose(params[1:], new_params[1:], rtol=0.03):
                print(f"Converged after {r} iterations")
                print(
                    f"Final parameters: dist={new_params[0]:.3f}m, poni1={new_params[1]/pixel_size:.3f}pix, poni2={new_params[2]/pixel_size:.3f}pix"
                )
                closest_y = (
                    round(new_params[1] / pixel_size) * pixel_size + pixel_size / 2
                )
                closest_x = (
                    round(new_params[2] / pixel_size) * pixel_size + pixel_size / 2
                )
                print(
                    f"Closest pixel coordinates: poni1={closest_y/pixel_size:.3f}pix, poni2={closest_x/pixel_size:.3f}pix"
                )
                return new_params
            else:
                params = new_params
                guessed_geom = Geometry(
                    dist=new_params[0],
                    poni1=new_params[1],
                    poni2=new_params[2],
                    detector=self.detector,
                    wavelength=wavelength,
                )
                r += 1
                print(
                    f"Step {r}: dist={params[0]:.3f}mm, poni1={params[1]/pixel_size:.3f}pix, poni2={params[2]/pixel_size:.3f}pix"
                )
