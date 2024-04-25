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
        self, exp, run, detector, geom, n_images=100, max_rings=5, pts_per_deg=1, I=0
    ):
        self.exp = exp
        self.run = run
        self.det_type = detector
        self.detector = self.load_geometry(geom)
        self.powder = self.load_powder_data()
        self.geometry = self.pyFAI_optimization(max_rings, pts_per_deg, I)

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
        calib_max = np.max(unassembled_images, axis=0)
        calib_max_flat = np.reshape(calib_max, (16 * 2 * 176, 2 * 192))
        return calib_max_flat

    def pyFAI_optimization(self, max_rings=5, pts_per_deg=1, I=0, max_iter=10):
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
        photon_energy = 1.23984197386209e-09 / wavelength
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
        best_score = +np.inf
        pixel_size = min(self.detector.pixel1, self.detector.pixel2)
        print(
            f"Starting optimization with initial guess: dist={params[0]:.3f}m, poni1={params[1]/pixel_size:.3f}pix, poni2={params[2]/pixel_size:.3f}pix"
        )
        sg = SingleGeometry(
            label="AgBh",
            image=self.powder,
            calibrant=behenate,
            detector=self.detector,
            geometry=guessed_geom,
        )
        for r in range(max_iter):
            sg.extract_cp(
                max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=I * photon_energy
            )
            score = sg.geometry_refinement.refine3(
                fix=["dist", "rot1", "rot2", "rot3", "wavelength"]
            )
            new_params = [
                sg.geometry_refinement.param[1],
                sg.geometry_refinement.param[2],
            ]
            if score < best_score:
                best_params = new_params
                best_score = score
                sg.geometry_refinement.poni1 = best_params[0]
                sg.geometry_refinement.poni2 = best_params[1]
            print(
                f"Step {r}: best poni1={params[1]/pixel_size:.3f}pix, best poni2={params[2]/pixel_size:.3f}pix"
            )
        return best_params, best_score
