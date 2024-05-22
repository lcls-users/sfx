import sys
import numpy as np
import matplotlib.pyplot as plt
import pyFAI
from pyFAI.calibrant import CalibrantFactory, CALIBRANT_FACTORY
from pyFAI.goniometer import SingleGeometry
from pyFAI.geometry import Geometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.gui import jupyter
from btx.interfaces.ipsana import *
from btx.diagnostics.run import RunDiagnostics
from btx.misc.metrology import *
from PSCalib.GeometryAccess import GeometryAccess

class PyFAIGeomOpt:
    """
    Class to perform Geometry Optimization using pyFAI

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    det_type : str
        Detector type
    detector : PyFAI(Detector)
        PyFAI detector object
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.distance = None
        self.poni1 = None
        self.poni2 = None
        self.detector = detector

    def pyFAI_geom_opt(self, powder, mask=None, max_rings=5, pts_per_deg=1, I=0, plot=''):
        """
        From guessed initial geometry, optimize the geometry using pyFAI package

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        max_rings : int
            Maximum number of rings to use for calibration
        pts_per_deg : int
            Number of points per degree to use for calibration (spacing of control points)
        I : int
            Set Minimum intensity to use for calibration based on photon energy
        plot : str
            Path to save plot of optimized geometry
        """

        if type(powder) == str:
            print(f"Loading powder {powder}")
            powder_img = np.load(powder)
        elif type(powder) == int:
            print("Computing powder from scratch")
            self.diagnostics.psi.calibrate = True
            powder_imgs = self.diagnostics.psi.get_images(powder, assemble=False)
            powder_img = np.max(powder_imgs, axis=0)
            powder_img = np.reshape(powder_img, self.detector.shape)
            if mask:
                print(f"Loading mask {mask}")
                mask = np.load(mask)
            else:
                mask = self.diagnostics.psi.get_mask()
            powder_img *= mask
        else:
            sys.exit("Unrecognized powder type, expected a path or number")
        self.img_shape = powder_img.shape

        print("Defining calibrant to be Silver Behenate...")
        behenate = CALIBRANT_FACTORY("AgBh")
        wavelength = self.diagnostics.psi.get_wavelength() * 1e-10
        photon_energy = 1.23984197386209e-09 / wavelength
        behenate.wavelength = wavelength

        print("Setting initial geometry guess...")
        if self.distance is None:
            distance = self.diagnostics.psi.estimate_distance() * 1e-3  # convert from mm to m
        if self.poni1 is None or self.poni2 is None:
            poni1 = 0   # assuming beam center is close to detector center
            poni2 = 0 
        guessed_geom = Geometry(
            dist=distance,
            poni1=poni1,
            poni2=poni2,
            detector=self.detector,
            wavelength=wavelength,
        )

        pixel_size = min(self.detector.pixel1, self.detector.pixel2)
        print(
            f"Starting optimization with initial guess: dist={self.distance:.3f}m, poni1={self.poni1/pixel_size:.3f}pix, poni2={self.poni2/pixel_size:.3f}pix"
        )
        sg = SingleGeometry(
            label="AgBh",
            image=powder_img,
            calibrant=behenate,
            detector=self.detector,
            geometry=guessed_geom,
        )
        sg.extract_cp(
            max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=I * photon_energy
        )
        score = sg.geometry_refinement.refine3(
            fix=["rot1", "rot2", "rot3", "wavelength"]
        )
        print(
            f"Optimization complete with final parameters: dist={distance:.3f}m, poni1={poni1/pixel_size:.3f}pix, poni2={poni2/pixel_size:.3f}pix"
        )
        self.distance = sg.geometry_refinement.param[0]
        self.poni1 = sg.geometry_refinement.param[1]
        self.poni2 = sg.geometry_refinement.param[2]

        if plot is not None:
            self.visualize_results()
        return score

    def deploy_geometry(self, outdir, psana_file=None, pv_camera_length=None):
        """
        Write new geometry files (.geom and .data for CrystFEL and psana respectively)
        with the optimized center and distance.

        Parameters
        ----------
        outdir : str
            path to output directory
        geom_init : str
            Initial geometry file in psana format
        pv_camera_length : str
            PV associated with camera length
        """
        # retrieve original geometry
        if psana_file is None:
            run = self.diagnostics.psi.run
            geom = self.diagnostics.psi.det.geometry(run)
        else:
            run = self.diagnostics.psi.run
            geom = GeometryAccess(psana_file)
        top = geom.get_top_geo()
        children = top.get_list_of_children()[0]

        # determine and deploy shifts in x,y,z
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        d1 = self.poni1 * 1e6
        d2 = self.poni2 * 1e6
        dz = self.distance * 1e6 - np.mean(p3)  # convert from m to microns
        geom.move_geo(children.oname, 0, dx=-d1, dy=-d2, dz=-dz)  # move the detector in psana frame
        self.geom = geom

        # write optimized geometry files
        psana_file, crystfel_file = os.path.join(outdir, f"r{run:04}_end.data"), os.path.join(outdir, f"r{run:04}.geom")
        temp_file = os.path.join(outdir, "temp.geom")
        geom.save_pars_in_file(psana_file)
        geometry_to_crystfel(psana_file, temp_file, cframe=0, zcorr_um=None)
        modify_crystfel_header(temp_file, crystfel_file)
        os.remove(temp_file)

        # Rayonix check
        if self.diagnostics.psi.get_pixel_size() != self.diagnostics.psi.det.pixel_size(
            run
        ):
            print(
                "Original geometry is wrong due to hardcoded Rayonix pixel size. Correcting geom file now..."
            )
            coffset = self.distance - self.diagnostics.psi.get_camera_length(pv_camera_length) / 1e3  # convert from mm to m
            res = 1e3 / self.diagnostics.psi.get_pixel_size()  # convert from mm to um
            os.rename(crystfel_file, temp_file)
            modify_crystfel_coffset_res(temp_file, crystfel_file, coffset, res)
            os.remove(temp_file)

    def display_panel(img=None, cp=None, ai=None, label=None, sg=None, ax=None):
        """Display an image with the control points and the calibrated rings
        in Jupyter notebooks

        Parameters
        ----------
        :param img: 2D numpy array with an image
        :param cp: ControlPoint instance
        :param ai: azimuthal integrator for iso-2th curves
        :param label: name of the curve
        :param sg: single geometry object regrouping img, cp and ai
        :param ax: subplot object to display in, if None, a new one is created.
        :param photon_energy: photon energy in eV
        :return: Matplotlib subplot
        """
        import numpy
        from matplotlib import lines
        from matplotlib.colors import SymLogNorm
        
        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
        if sg is not None:
            if img is None:
                img = sg.image
            if cp is None:
                cp = sg.control_points
            if ai is None:
                ai = sg.geometry_refinement
            if label is None:
                label = sg.label
        n_panels = [2, 6, 10, 14]
        for i in range(len(n_panels)):
            img_panel = img[n_panels[i]*352:(n_panels[i]+1)*352,:]
            ax[i//2, i%2].imshow(img_panel,
                    origin="lower",
                    cmap="viridis",
                    vmax=np.max(img)/100)
            label = f'panel {n_panels[i]}'
            ax[i//2, i%2].set_title(label, fontsize='small')
            if cp is not None:
                for lbl in cp.get_labels():
                    pt = numpy.array(cp.get(lbl=lbl).points)
                    pt_lbl = []
                    if len(pt) > 0:
                        for point in pt:
                            if n_panels[i]*352 <= point[0] <= (n_panels[i]+1)*352:
                                pt_lbl.append(point)
                        pt_lbl = np.array(pt_lbl)
                        if len(pt_lbl) > 0:
                            ax[i//2, i%2].scatter(pt_lbl[:,1], pt_lbl[:,0]-n_panels[i]*352, label=lbl, s=10)
            if ai is not None and cp.calibrant is not None:
                tth = cp.calibrant.get_2th()
                ttha = ai.twoThetaArray()
                ax[i//2, i%2].contour(ttha[n_panels[i]*352:(n_panels[i]+1)*352,:], levels=tth, cmap="autumn", linewidths=2, linestyles="dashed")
            ax[i//2, i%2].axis('off')
        return ax

    def visualize_results(self, flat_powder, sg0=None, sg1=None, plot=''):
        """
        Visualize the extraction of control points and the radial profiles
        before and after optimization

        Parameters
        ----------
        flat_powder : numpy.ndarray
            Powder image
        sg0 : SingleGeometry
            SingleGeometry object before optimization
        sg1 : SingleGeometry
            SingleGeometry object after optimization
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(10, 8), dpi=300)

        subfigs = fig.subfigures(2, 2, height_ratios=[1.2, 1])

        # plotting control points extraction
        subfigs[0, 0].suptitle('Powder before Optimization')
        ax0 = subfigs[0, 0].subplots(2, 2)
        self.display_panel(sg=sg0, ax=ax0)
        subfigs[0, 1].suptitle('Powder after Optimization')
        ax1 = subfigs[0, 1].subplots(2, 2)
        self.display_panel(sg=sg1, ax=ax1)

        # plotting radial profiles with peaks
        subfigs[1, 0].suptitle('Radially averaged intensity before Optimization')
        ax2 = subfigs[1, 0].subplots()
        ai = sg0.geometry_refinement
        rai = ai.integrate1d(flat_powder, 1000)
        jupyter.plot1d(rai, calibrant=sg0.calibrant, ax=ax2)
        subfigs[1, 1].suptitle('Radially averaged intensity after Optimization')
        ax3 = subfigs[1, 1].subplots()
        ai = sg1.geometry_refinement
        rai = ai.integrate1d(flat_powder, 1000)
        jupyter.plot1d(rai, calibrant=sg1.calibrant, ax=ax3)

        plt.tight_layout()
        if plot != '':
            fig.savefig(plot, dpi=300)
