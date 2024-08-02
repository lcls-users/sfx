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
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import os
import csv
import yaml
import copy
import shutil
import random

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
        calibrant,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant

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

        print("Defining calibrant...")
        behenate = CALIBRANT_FACTORY(self.calibrant)
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

class BayesGeomOpt:
    """
    Class to perform Geometry Optimization using Bayesian Optimization on pyFAI

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
    calibrant : str
        Calibrant name
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.PARAM_ORDER = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"]
        self.DEFAULT_VALUE = {"dist":self.diagnostics.psi.estimate_distance() * 1e-3, "poni1":0, "poni2":0, "rot1":0, "rot2":0, "rot3":0, "wavelength":self.diagnostics.psi.get_wavelength() * 1e-10}
        self.DIST_RES = 0.001
        self.PONI_RES = 0.0001
        self.ROT_RES = 0.001

    @staticmethod
    def expected_improvement(X, gp_model, best_y):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y) / y_std
        ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
        return ei

    @staticmethod
    def upper_confidence_bound(X, gp_model, beta):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        return ucb

    @staticmethod
    def probability_of_improvement(X, gp_model, best_y):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y) / y_std
        pi = norm.cdf(z)
        return pi
    
    def bayesian_geom_opt(
        self,
        powder,
        fix,
        bounds,
        mask=None,
        n_samples=10,
        num_iterations=100,
        af="ucb",
        seed=0,
    ):
        """
        From guessed initial geometry, optimize the geometry using Bayesian Optimization on pyFAI package

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        fix : list
            List of parameters not to be optimized
        bounds : dict
            Dictionary of bounds for each parameter
        n_samples : int
            Number of samples to initialize the GP model
        num_iterations : int
            Number of iterations for optimization
        af : str
            Acquisition function to use for optimization
        seed : int
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
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

        print("Defining calibrant...")
        calibrant = CALIBRANT_FACTORY(self.calibrant)
        wavelength = self.DEFAULT_VALUE["wavelength"]
        photon_energy = 1.23984197386209e-09 / wavelength
        calibrant.wavelength = wavelength
        
        print("Setting geometry space...")
        bo_history = {}
        np.random.seed(seed)
        input_range = {}
        input_range_norm = {}
        for param in self.PARAM_ORDER:
            print(f"Setting space for {param}...")
            if param in fix:
                input_range[param] = np.array([self.DEFAULT_VALUE[param]])
                input_range_norm[param] = np.array([1])
            elif param == "dist":
                input_range[param] = np.arange(bounds[param][0], bounds[param][1]+self.DIST_RES, self.DIST_RES)
                input_range_norm[param] = (input_range[param]-np.min(input_range[param]))/(np.max(input_range[param])-np.min(input_range[param]))
            elif param in ["poni1", "poni2"]:
                input_range[param] = np.arange(bounds[param][0], bounds[param][1]+self.PONI_RES, self.PONI_RES)
                input_range_norm[param] = (input_range[param]-np.min(input_range[param]))/(np.max(input_range[param])-np.min(input_range[param]))
            else:
                input_range[param] = np.arange(bounds[param][0], bounds[param][1]+self.ROT_RES, self.ROT_RES)
                input_range_norm[param] = (input_range[param]-np.min(input_range[param]))/(np.max(input_range[param])-np.min(input_range[param]))
        X = np.array(np.meshgrid(*[input_range[param] for param in self.PARAM_ORDER])).T.reshape(-1, len(self.PARAM_ORDER))
        X_norm = np.array(np.meshgrid(*[input_range_norm[param] for param in self.PARAM_ORDER])).T.reshape(-1, len(self.PARAM_ORDER))
        print(f"Setting space complete with {X.shape[0]} points")
        idx_samples = np.random.choice(X.shape[0], n_samples)
        X_samples = X[idx_samples]
        X_norm_samples = X_norm[idx_samples]
        y = np.zeros((n_samples, 1))

        print("Initializing samples...")
        for i in range(n_samples):
            print(f"Initializing sample {i+1}...")
            dist, poni1, poni2, rot1, rot2, rot3, wavelength = X_samples[i]
            geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
            score = sg.geometry_refinement.refine3(fix=["wavelength"])
            y[i] = score
            bo_history[f'init_sample_{i+1}'] = {'param':X[i], 'score': score}

        kernel = RBF(length_scale=0.3, length_scale_bounds='fixed') \
                * ConstantKernel(constant_value=1.0, constant_value_bounds=(0.5, 1.5)) \
                + WhiteKernel(noise_level=0.001, noise_level_bounds = 'fixed')
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=seed)
        gp_model.fit(X_norm_samples, y)
        visited_idx = list(idx_samples.flatten())

        if af == "ucb":
            beta = 0.6
            af = self.upper_confidence_bound
        elif af == "ei":
            af = self.expected_improvement
        elif af == "pi":
            af = self.probability_of_improvement

        print("Starting Bayesian optimization...")
        for i in range(num_iterations):
            print(f"Iteration {i+1}...")
            # 1. Generate the Acquisition Function values using the Gaussian Process Regressor
            af_values = af(X_norm, gp_model, beta)
            af_values[visited_idx] = -np.inf
            
            # 2. Select the next set of parameters based on the Acquisition Function
            new_idx = np.argmax(af_values)
            new_input = X[new_idx]
            visited_idx.append(new_idx)

            # 3. Compute the score of the new set of parameters 
            dist, poni1, poni2, rot1, rot2, rot3, wavelength = new_input
            geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
            score = sg.geometry_refinement.refine3(fix=["wavelength"])
            bo_history[f'iteration_{i+1}'] = {'param':X[new_idx], 'score': score}
            y = np.append(y, score)
            X_samples = np.append(X_samples, X[new_idx])
            X_norm_samples = np.append(X_norm_samples, X_norm[new_idx])

            # 4. Update the Gaussian Process Regressor
            gp_model.fit(X_norm_samples, y)
        
        best_idx = np.argmax(y)
        best_param = X_samples[best_idx]
        best_score = y[best_idx]
        return bo_history, best_param, best_score
    
    def convergence_plot(bo_history):
        """
        Plot the convergence of the objective function over time

        Parameters
        ----------
        bo_history : dict
            Dictionary containing the history of optimization
        """
        scores = [bo_history[key]['score'] for key in bo_history.keys()]
        plt.plot(scores)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Convergence Plot')
        plt.show()

class HookeJeevesGeomOpt:
    """
    Class to perform Geometry Optimization using Hooke-Jeeves algorithm on pyFAI
    
    Note: this is intented to work only for 3D optimization

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
    calibrant : str
        Calibrant name
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.PARAM_ORDER = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"]
        self.DEFAULT_VALUE = {"dist":self.diagnostics.psi.estimate_distance() * 1e-3, "poni1":0, "poni2":0, "rot1":0, "rot2":0, "rot3":0, "wavelength":self.diagnostics.psi.get_wavelength() * 1e-10}

    def hookes_jeeves_geom_opt(
        self,
        powder,
        fix,
        mask=None,
        step_size=0.01,
        tol=0.00001, 
    ):
        """
        From guessed initial geometry, optimize the geometry using Hooke-Jeeves algorithm on pyFAI package

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        fix : list
            List of parameters not to be optimized
        mask : str
            Path to mask file
        step_size : float
            Initial tep size for optimization
        tol : float
            Tolerance for convergence
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

        print("Defining calibrant...")
        calibrant = CALIBRANT_FACTORY(self.calibrant)
        wavelength = self.diagnostics.psi.get_wavelength() * 1e-10
        photon_energy = 1.23984197386209e-09 / wavelength
        calibrant.wavelength = wavelength
        
        hjo_history = {}
        x = np.array([self.DEFAULT_VALUE[param] for param in self.PARAM_ORDER])
        scores = []
        dist, poni1, poni2, rot1, rot2, rot3, wavelength = x
        geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
        sg = SingleGeometry("extract_cp", powder, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
        sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
        score = sg.geometry_refinement.refine3(fix=fix)
        n = 0
        scores.append(score)
        hjo_history[f'iteration_{n}'] = {'param':x, 'score': score}
        while step_size >= tol:
            neighbours = {}
            for param in self.PARAM_ORDER:
                if param not in fix:
                    i = self.PARAM_ORDER.index(param)
                    x_plus = x.copy()
                    x_plus[i] += step_size
                    x_minus = x.copy()
                    x_minus[i] -= step_size
                    neighbours[f'x_plus_{param}'] = x_plus
                    neighbours[f'x_minus_{param}'] = x_minus
            for key in neighbours.keys():
                dist, poni1, poni2, rot1, rot2, rot3, wavelength = neighbours[key]
                geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
                sg = SingleGeometry("extract_cp", powder, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
                sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
                score = sg.geometry_refinement.refine3(fix=["wavelength"])
                scores.append(score)
            best_idx = np.argmin(scores)
            if best_idx == 0:
                step_size /= 2
            else:
                x = neighbours[f'x_plus_{params[best_idx]}'] if best_idx % 2 == 0 else neighbours[f'x_minus_{params[best_idx]}']
                n += 1
                hjo_history[f'iteration_{n}'] = {'param':x, 'score': scores[best_idx]}
        return hjo_history, x, scores[best_idx]


