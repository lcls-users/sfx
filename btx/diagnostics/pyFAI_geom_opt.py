import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyFAI
from pyFAI.calibrant import CalibrantFactory, CALIBRANT_FACTORY
from pyFAI.goniometer import SingleGeometry
from pyFAI.geometry import Geometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.gui import jupyter
from btx.interfaces.ipsana import *
from btx.diagnostics.run import RunDiagnostics
from btx.misc.metrology import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared
from scipy.stats import norm
from tqdm import tqdm

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

class GridSearchGeomOpt:
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
    Imin : int or str
        Minimum intensity to use for control point extraction based on photon energy or max intensity
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
        Imin,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.Imin = Imin

    def grid_search_geom_opt(
        self,
        powder,
        bounds,
        dist,
        mask=None,
    ):
        """
        From guessed initial geometry, optimize the geometry using Grid Search algorithm on pyFAI package, only for poni1 and poni2

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        bounds : dict
            Dictionary of bounds for each parameter
        dist : float
            Distance to sample
        mask : str
            Path to mask file
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

        print("Setting minimal intensity...")
        if type(self.Imin) == str:
            Imin = np.max(powder_img) * 0.01
        else:
            Imin = self.Imin * photon_energy

        print("Setting geometry space...")
        poni1_range = np.linspace(bounds["poni1"][0], bounds["poni1"][1], bounds["poni1"][2])
        poni2_range = np.linspace(bounds["poni2"][0], bounds["poni2"][1], bounds["poni2"][2])
        cy, cx = np.meshgrid(poni1_range, poni2_range)
        y = np.zeros_like(cx)
        for i in tqdm(range(len(poni1_range))):
            for j in range(len(poni2_range)):
                    poni1 = cx[i, j]
                    poni2 = cy[i, j]
                    geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, detector=self.detector, wavelength=wavelength)
                    sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
                    sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
                    if len(sg.geometry_refinement.data) == 0:
                        y[i, j] = 0
                    else:
                        y[i, j] = len(sg.geometry_refinement.data)
        return cx, cy, y

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
    Imin : int
        Minimum intensity to use for control point extraction based on photon energy
    fix : list
        List of parameters not to be optimized
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
        fix,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.fix = fix
        self.param_order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.param_space = []
        for param in self.param_order:
            if param not in fix:
                self.param_space.append(param)
        self.distance = self.diagnostics.psi.estimate_distance() * 1e-3
        self.cx, self.cy, _ = detector.calc_cartesian_positions()
        self.default_value = {'dist':self.distance, 'poni1':self.cx, 'poni2':self.cy}

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
        Imin,
        bounds,
        values,
        n_samples,
        num_iterations,
        af="ei",
        mask=None,
        seed=0,
    ):
        """
        From guessed initial geometry, optimize the geometry using Bayesian Optimization on pyFAI package

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        fix : list
            List of parameters not to be optimized in refine3
            Nota Bene: this fix could be different from self.fix
        Imin : int or str
            Minimum intensity to use for control point extraction based on photon energy or max intensity
        bounds : dict
            Dictionary of bounds and resolution for search parameters
        values : dict
            Dictionary of values for fixed parameters
        n_samples : int
            Number of samples to initialize the GP model
        num_iterations : int
            Number of iterations for optimization
        af : str
            Acquisition function to use for optimization
        mask : np.ndarray
            Mask for powder image
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        if type(powder) == str:
            print(f"Loading powder {powder}")
            powder_img = np.load(powder)
        elif type(powder) == int:
            print("Computing powder from scratch")
            self.diagnostics.psi.counter = 0
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
        
        print("Defining minimal intensity...")
        if type(Imin) == str:
            Imin = np.max(powder_img) * 0.01
        else:
            Imin = Imin * photon_energy

        print("Defining geometry parameter space...")
        print(f"Search space: {self.param_space}")
        bo_history = {}
        input_range = {}
        input_range_norm = {}
        for param in self.param_order:
            if param in self.param_space:
                print(f"Defining search space for {param}...")
                input_range[param] = np.arange(bounds[param][0], bounds[param][1]+bounds[param][2], bounds[param][2])
                input_range_norm[param] = input_range[param]
            else:
                if values[param] is None:
                    print(f"Fixed parameter {param} with value {self.default_value[param]}")
                    input_range[param] = np.array([self.default_value[param]])
                else:
                    print(f'Fixed parameter {param} with value {values[param]}')
                    input_range[param] = np.array([values[param]])
        X = np.array(np.meshgrid(*[input_range[param] for param in self.param_order])).T.reshape(-1, len(self.param_order))
        X_norm = np.array(np.meshgrid(*[input_range_norm[param] for param in self.param_space])).T.reshape(-1, len(self.param_space))
        X_norm = (X_norm - np.mean(X_norm, axis=0)) / np.std(X_norm, axis=0)
        print(f"Search space: {X_norm.shape[0]} points")
        idx_samples = np.random.choice(X.shape[0], n_samples)
        X_samples = X[idx_samples]
        X_norm_samples = X_norm[idx_samples]
        y = np.zeros((n_samples))

        print("Initializing samples...")
        for i in range(n_samples):
            print(f"Initializing sample {i+1}...")
            dist, poni1, poni2, rot1, rot2, rot3 = X_samples[i]
            geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
            y[i] = len(sg.geometry_refinement.data)
            bo_history[f'init_sample_{i+1}'] = {'param':X_samples[i], 'score': y[i]}

        print("Standardizing initial score values...")
        y_norm = (y - np.mean(y)) / np.std(y)

        print("Fitting Gaussian Process Regressor...")
        kernel = RBF(length_scale=1.0, length_scale_bounds='fixed') \
                * ConstantKernel(constant_value=1.0, constant_value_bounds=(0.1, 10.0)) \
                + WhiteKernel(noise_level=0.001, noise_level_bounds = 'fixed')
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list(idx_samples.flatten())

        if af == "ucb":
            beta = 50
            af = self.upper_confidence_bound
        elif af == "ei":
            af = self.expected_improvement
        elif af == "pi":
            af = self.probability_of_improvement

        print("Starting Bayesian optimization...")
        for i in tqdm(range(num_iterations)):
            # 1. Generate the Acquisition Function values using the Gaussian Process Regressor
            af_values = af(X_norm, gp_model, best_score)
            af_values[visited_idx] = -np.inf
            
            # 2. Select the next set of parameters based on the Acquisition Function
            new_idx = np.argmax(af_values)
            new_input = X[new_idx]
            visited_idx.append(new_idx)

            # 3. Compute the score of the new set of parameters
            dist, poni1, poni2, rot1, rot2, rot3 = new_input
            geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
            score = len(sg.geometry_refinement.data)
            y = np.append(y, [score], axis=0)
            bo_history[f'iteration_{i+1}'] = {'param':X[new_idx], 'score': score}
            X_samples = np.append(X_samples, [X[new_idx]], axis=0)
            X_norm_samples = np.append(X_norm_samples, [X_norm[new_idx]], axis=0)
            y_norm = (y - np.mean(y)) / np.std(y)
            best_score = np.max(y_norm)
            # 4. Update the Gaussian Process Regressor
            gp_model.fit(X_norm_samples, y_norm)
        
        best_idx = np.argmax(y_norm)
        best_param = X_samples[best_idx]
        dist, poni1, poni2, rot1, rot2, rot3 = best_param
        geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
        sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
        sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
        residual = sg.geometry_refinement.refine3(fix=fix)
        self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 = sg.geometry_refinement.param
        return bo_history, best_idx, residual

    def grid_search_convergence_plot(self, bo_history, bounds, best_idx, grid_search, plot):
        """
        Returns an animation of the Bayesian Optimization iterations on the grid search space

        Parameters
        ----------
        bo_history : dict
            Dictionary containing the history of optimization
        bounds : dict
            Dictionary of bounds for each parameter
        best_param : np.ndarray
            Best parameters found
        grid_search : np.ndarray
            Grid search space
        plot : str
            Path to save plot
        """
        scores = [bo_history[key]['score'] for key in bo_history.keys()]
        params = [bo_history[key]['param'] for key in bo_history.keys()]
        if len(self.param_space)==1:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].plot([param[0] for param in params])
            ax[0].set_xticks(np.arange(len(scores), step=5))
            ax[0].set_xlabel('Iteration')
            ax[0].set_ylabel('Distance (m)')
            ax[0].set_title('Bayesian Optimization on Detector Distance')
            best_param = params[best_idx]
            ax[0].axvline(x=best_idx, color='red', linestyle='dashed')
            ax[0].axhline(y=best_param[0], color='green', linestyle='dashed')
            ax[1].plot(np.maximum.accumulate(scores))
            ax[1].set_xticks(np.arange(len(scores), step=5))
            ax[1].set_xlabel('Iteration')
            ax[1].set_ylabel('Best score so far')
            ax[1].set_title('Convergence Plot')
            ax[1].axvline(x=best_idx, color='red', linestyle='dashed')
            ax[1].axhline(y=scores[best_idx], color='green', linestyle='dashed')
        elif len(self.param_space)==2:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            poni1_range = np.linspace(bounds['poni1'][0], bounds['poni1'][1], grid_search.shape[0])
            poni2_range = np.linspace(bounds['poni2'][0], bounds['poni2'][1], grid_search.shape[1])
            cy, cx = np.meshgrid(poni1_range, poni2_range)
            c = ax[0].pcolormesh(cx, cy, grid_search, cmap='RdBu')
            ax[0].set_xlabel('Poni1 (m)')
            ax[0].set_ylabel('Poni2 (m)')
            ax[0].set_aspect('equal')
            ax[0].set_title('Bayesian Optimization Convergence on Grid Search Space')
            ax[0].scatter([param[1] for param in params], [param[2] for param in params], c=np.arange(len(params)), cmap='RdYlGn')
            cbar = plt.colorbar(c, ax=ax[0])
            cbar.set_label('Score')
            best_param = params[best_idx]
            ax[0].scatter(best_param[1], best_param[2], c='white', s=100, label='best', alpha=0.3)
            ax[0].legend()
            ax[1].plot(np.maximum.accumulate(scores))
            ax[1].set_xticks(np.arange(len(scores), step=5))
            ax[1].set_xlabel('Iteration')
            ax[1].set_ylabel('Best score so far')
            ax[1].set_title('Convergence Plot')
            ax[1].axvline(x=best_idx, color='red', linestyle='dashed')
            ax[1].axhline(y=scores[best_idx], color='green', linestyle='dashed')
        elif len(self.param_space)==3:
            fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            poni1_range = np.linspace(bounds['poni1'][0], bounds['poni1'][1], grid_search.shape[0])
            poni2_range = np.linspace(bounds['poni2'][0], bounds['poni2'][1], grid_search.shape[1])
            cy, cx = np.meshgrid(poni1_range, poni2_range)
            c = ax[0].pcolormesh(cx, cy, grid_search, cmap='RdBu')
            ax[0].set_xlabel('Poni1 (m)')
            ax[0].set_ylabel('Poni2 (m)')
            ax[0].set_aspect('equal')
            ax[0].set_title('Bayesian Optimization Convergence on Grid Search Space')
            ax[0].scatter([param[1] for param in params], [param[2] for param in params], c=np.arange(len(params)), cmap='RdYlGn')
            cbar = plt.colorbar(c, ax=ax[0])
            cbar.set_label('Score')
            best_param = params[best_idx]
            ax[0].scatter(best_param[1], best_param[2], c='white', s=100, label='best', alpha=0.3)
            ax[0].legend()
            ax[1].plot([param[0] for param in params])
            ax[1].set_xticks(np.arange(len(scores), step=5))
            ax[1].set_xlabel('Iteration')
            ax[1].set_ylabel('Distance (m)')
            ax[1].set_title('Bayesian Optimization on Detector Distance')
            ax[1].axvline(x=best_idx, color='red', linestyle='dashed')
            ax[1].axhline(y=best_param[0], color='green', linestyle='dashed')
            ax[2].plot(np.maximum.accumulate(scores))
            ax[2].set_xticks(np.arange(len(scores), step=5))
            ax[2].set_xlabel('Iteration')
            ax[2].set_ylabel('Best score so far')
            ax[2].set_title('Convergence Plot')
            ax[2].axvline(x=best_idx, color='red', linestyle='dashed')
            ax[2].axhline(y=scores[best_idx], color='green', linestyle='dashed')
        fig.tight_layout()
        fig.savefig(plot)

class HookeJeevesGeomOpt:
    """
    Class to perform Geometry Optimization using Hooke-Jeeves algorithm on pyFAI

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
    Imin : int
        Minimum intensity to use for control point extraction based on photon energy
    fix : list
        List of parameters not to be optimized
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
        Imin,
        fix,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.Imin = Imin
        self.fix = fix
        self.param_order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.default_value = [self.diagnostics.psi.estimate_distance() * 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.param_space = []
        for param in self.param_order:
            if param not in fix:
                self.param_space.append(param)

    def hooke_jeeves_geom_opt(
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
            List of parameters not to be optimized in refine3
            Nota Bene: this fix could be different from self.fix
        bounds : dict
            Dictionary of bounds for each parameter
        mask : str
            Path to mask file
        step_size : float
            Initial step size for optimization
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
        
        print("Setting minimal intensity...")
        if type(self.Imin) == str:
            Imin = np.max(powder_img) * 0.01
        else:
            Imin = self.Imin * photon_energy

        hjo_history = {}
        dist, poni1, poni2, rot1, rot2, rot3 = self.default_value
        x = np.array(self.default_value)
        geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
        sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
        sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=self.Imin*photon_energy)
        score = sg.geometry_refinement.refine3(fix=fix)
        i = 0
        hjo_history[f'iteration_{i+1}'] = {'param':x, 'optim': sg.geometry_refinement.param, 'score': score}
        while step_size >= tol:
            print(f"Iteration {i+1}...")
            scores = []
            neighbours = []
            for param in self.param_space:
                j = self.param_order.index(param)
                x_plus = x.copy()
                x_plus[j] += step_size
                x_minus = x.copy()
                x_minus[j] -= step_size
                neighbours.append(x_plus)
                neighbours.append(x_minus)
            for neighbour in neighbours:
                dist, poni1, poni2, rot1, rot2, rot3 = neighbour
                geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
                sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
                sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
                score = sg.geometry_refinement.refine3(fix=fix)
                scores.append(score)
            best_idx = np.argmin(scores)
            if best_idx == 0:
                step_size /= 10
                print(f"Reducing step size to {step_size}")
            else:
                x = neighbours[best_idx]
                i += 1
                hjo_history[f'iteration_{i+1}'] = {'param':x, 'score': scores[best_idx]}

            dist, poni1, poni2, rot1, rot2, rot3 = x
            best_idx = np.argmin(scores)
            geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
            best_score = sg.geometry_refinement.refine3(fix=fix)
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 = sg.geometry_refinement.param
        return hjo_history, best_idx, best_score
    
    def grid_search_convergence_plot(self, hj_history, best_idx, grid_search, plot):
        """
        Returns an animation of the Bayesian Optimization iterations on the grid search space

        Parameters
        ----------
        bo_history : dict
            Dictionary containing the history of optimization
        best_param : np.ndarray
            Best parameters found
        grid_search : np.ndarray
            Grid search space
        plot : str
            Path to save plot
        """
        scores = [hj_history[key]['score'] for key in hj_history.keys()]
        params = [hj_history[key]['param'] for key in hj_history.keys()]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        poni1_range = np.linspace(-0.01, 0.01, 51)
        poni2_range = np.linspace(-0.01, 0.01, 51)
        cy, cx = np.meshgrid(poni1_range, poni2_range)
        ax[0].pcolormesh(cx, cy, grid_search, cmap='RdBu')
        ax[0].set_xlabel('Poni1')
        ax[0].set_ylabel('Poni2')
        ax[0].set_aspect('equal')
        ax[0].set_title('Hooke-Jeeves Search Convergence on Grid Search Space')
        ax[0].scatter([param[1] for param in params], [param[2] for param in params], c=np.arange(len(params)), cmap='RdYlGn')
        best_param = params[best_idx]
        ax[0].scatter(best_param[1], best_param[2], c='red', s=100, label='best', alpha=0.3)
        ax[0].legend()
        ax[1].plot(scores)
        ax[1].set_xticks(np.arange(len(scores), step=5))
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Score')
        ax[1].set_yscale('log')
        ax[1].set_aspect('auto')
        ax[1].set_title('Convergence Plot')
        ax[1].axvline(x=best_idx, color='red', linestyle='dashed')
        ax[1].axhline(y=scores[best_idx], color='green', linestyle='dashed')
        fig.tight_layout()
        fig.savefig(plot)

class CrossEntropyGeomOpt:
    """
    Class to perform Geometry Optimization using Cross-Entropy algorithm on pyFAI

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
    Imin : int
        Minimum intensity to use for control point extraction based on photon energy
    fix : list
        List of parameters not to be optimized
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
        Imin,
        fix,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.Imin = Imin
        self.fix = fix
        self.param_order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.default_value = [self.diagnostics.psi.estimate_distance() * 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.param_space = []
        for param in self.param_order:
            if param not in fix:
                self.param_space.append(param)        

    def cross_entropy_geom_opt(
        self,
        powder,
        fix,
        bounds,
        mask=None,
        n_samples=100,
        m_elite=10,
        num_iterations=100,
        seed=0,
    ):
        """
        From guessed initial geometry, optimize the geometry using Cross-Entropy algorithm on pyFAI package

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        fix : list
            List of parameters not to be optimized
        bounds : dict
            Dictionary of bounds for each parameter
        mask : str
            Path to mask file
        n_samples : int
            Number of samples to initialize the refit the distribution
        m_elite : int
            Number of elite samples to keep in the population for each generation
        num_iterations : int
            Number of iterations for optimization
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
        wavelength = self.diagnostics.psi.get_wavelength() * 1e-10
        photon_energy = 1.23984197386209e-09 / wavelength
        calibrant.wavelength = wavelength

        print("Setting minimal intensity...")
        if type(self.Imin) == str:
            Imin = np.max(powder_img) * 0.01
        else:
            Imin = self.Imin * photon_energy
        
        print("Setting geometry space...")
        print(f"Search space: {self.param_space}")
        ce_history = {}
        means = np.array([np.mean(bounds[param]) for param in self.param_space])
        cov = np.diag([np.std(bounds[param])**2 for param in self.param_space])
        for i in tqdm(range(num_iterations)):
            print(f"Iteration {i+1}...")
            X = np.random.multivariate_normal(means, cov, n_samples)
            X_samples = X.copy()
            for param in self.param_order:
                if param in self.fix:
                    idx = self.param_order.index(param)
                    X_samples = np.insert(X_samples, idx, self.default_value[idx], axis=1)
            scores = []
            for j in range(n_samples):
                dist, poni1, poni2, rot1, rot2, rot3 = X_samples[j]
                geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
                sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
                sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
                if len(sg.geometry_refinement.data) == 0:
                    score = np.inf
                else:
                    score = sg.geometry_refinement.refine3(fix=fix)
                scores.append(score)
                ce_history[f'iteration_{i+1}_sample_{j+1}'] = {'param':X_samples[j], 'score': score}
            elite_idx = np.argsort(scores)[:m_elite]
            elite_samples = X[elite_idx]
            means = np.mean(elite_samples, axis=0)
            cov = np.cov(elite_samples.T)
        best_idx = np.argmin([ce_history[key]['score'] for key in ce_history.keys()])
        q, r = divmod(best_idx, num_iterations)
        dist, poni1, poni2, rot1, rot2, rot3 = ce_history[f'iteration_{q+1}_sample_{r+1}']['param']
        geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
        sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
        sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=Imin)
        best_score = sg.geometry_refinement.refine3(fix=fix)
        self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 = sg.geometry_refinement.param
        return ce_history, best_idx, best_score
    
    def grid_search_convergence_plot(self, ce_history, best_idx, grid_search, plot):
        """
        Returns an animation of the Bayesian Optimization iterations on the grid search space

        Parameters
        ----------
        bo_history : dict
            Dictionary containing the history of optimization
        best_param : np.ndarray
            Best parameters found
        grid_search : np.ndarray
            Grid search space
        plot : str
            Path to save plot
        """
        scores = [ce_history[key]['score'] for key in ce_history.keys()]
        params = [ce_history[key]['param'] for key in ce_history.keys()]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        poni1_range = np.linspace(-0.01, 0.01, 51)
        poni2_range = np.linspace(-0.01, 0.01, 51)
        cy, cx = np.meshgrid(poni1_range, poni2_range)
        ax[0].pcolormesh(cx, cy, grid_search, cmap='RdBu')
        ax[0].set_xlabel('Poni1')
        ax[0].set_ylabel('Poni2')
        ax[0].set_aspect('equal')
        ax[0].set_title('Cross Entropy Convergence on Grid Search Space')
        ax[0].scatter([param[1] for param in params], [param[2] for param in params], c=np.arange(len(params)), cmap='RdYlGn')
        best_param = params[best_idx]
        ax[0].scatter(best_param[1], best_param[2], c='red', s=100, label='best', alpha=0.3)
        ax[0].legend()
        ax[1].plot(scores)
        ax[1].set_xticks(np.arange(len(scores), step=5))
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Score')
        ax[1].set_yscale('log')
        ax[1].set_aspect('auto')
        ax[1].set_title('Convergence Plot')
        ax[1].axvline(x=best_idx, color='red', linestyle='dashed')
        ax[1].axhline(y=scores[best_idx], color='green', linestyle='dashed')
        fig.tight_layout()
        fig.savefig(plot)

class SimulatedAnnealingGeomOpt:
    """
    Class to perform Geometry Optimization using Simulated Annealing algorithm on pyFAI

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
    fix : list
        List of parameters not to be optimized
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
        fix,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.fix = fix
        self.param_order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.default_value = [self.diagnostics.psi.estimate_distance() * 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.param_space = []
        for param in self.param_order:
            if param not in fix:
                self.param_space.append(param)

    @staticmethod
    def acceptance_probability(old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temperature)

    def simulated_annealing_geom_opt(
        self,
        powder,
        fix,
        mask=None,
        initial_temp=100,
        cooling_rate=0.01,
        num_iterations=100,
        eps=0.1,
        seed=0,
    ):
        """
        From guessed initial geometry, optimize the geometry using Simulated Annealing algorithm on pyFAI package

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        fix : list
            List of parameters not to be optimized
        mask : str
            Path to mask file
        initial_temp : float
            Initial temperature for optimization
        cooling_rate : float
            Cooling rate for the temperature
        num_iterations : int
            Number of iterations for optimization
        eps : float
            Perturbation factor for the parameters
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
        self.img_shape = powder_img.shape

        print("Defining calibrant...")
        calibrant = CALIBRANT_FACTORY(self.calibrant)
        wavelength = self.diagnostics.psi.get_wavelength() * 1e-10
        photon_energy = 1.23984197386209e-09 / wavelength
        calibrant.wavelength = wavelength
        
        sa_history = {}
        dist, poni1, poni2, rot1, rot2, rot3 = self.default_value
        x = np.array(self.default_value)
        geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
        sg = SingleGeometry("extract_cp", powder, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
        sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
        score = sg.geometry_refinement.refine3(fix=fix)
        best_param = x
        best_score = score
        temperature = initial_temp

        for i in range(num_iterations):
            print(f"Iteration {i+1}...")
            x_new = x.copy()
            idx = np.random.choice([self.param_order.index(param) for param in self.param_space])
            x_new[idx] += np.random.uniform(-eps, eps)
            dist, poni1, poni2, rot1, rot2, rot3 = x_new
            geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
            sg = SingleGeometry("extract_cp", powder, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
            new_score = sg.geometry_refinement.refine3(fix=fix)
            if self.acceptance_probability(score, new_score, temperature) > np.random.random():
                x = x_new
                score = new_score
                if new_score < best_score:
                    best_param = x_new
                    best_score = new_score
            temperature *= cooling_rate
            sa_history[f'iteration_{i+1}'] = {'param':x, 'score': score}
        return sa_history, best_param, best_score

class GeneticAlgGeomOpt:
    """
    Class to perform Geometry Optimization using Genetic Algorithm on pyFAI

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
    fix : list
        List of parameters not to be optimized
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
        fix,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.detector = detector
        self.calibrant = calibrant
        self.fix = fix
        self.param_order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.default_value = [self.diagnostics.psi.estimate_distance() * 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.param_space = []
        for param in self.param_order:
            if param not in fix:
                self.param_space.append(param)

    def genetic_alg_geom_opt(
        self,
        powder,
        fix, 
        bounds,
        sigma = 0.1,
        mask=None,
        n_samples=10,
        m_elite=5,
        num_iterations=10,
        seed=0,
    ):
        """
        From guessed initial geometry, optimize the geometry using Genetic Algorithm on pyFAI package

        Parameters
        ----------
        powder : str or int
            Path to powder image or number of images to use for calibration
        fix : list
            List of parameters not to be optimized
        bounds : dict
            Dictionary of bounds for each parameter
        sigma : float
            Standard deviation for the mutation
        mask : str
            Path to mask file
        n_samples : int
            Number of samples to initialize the population
        m_elite : int
            Number of elite samples to keep in the population for each generation
        num_iterations : int
            Number of iterations for optimization
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
        wavelength = self.diagnostics.psi.get_wavelength() * 1e-10
        photon_energy = 1.23984197386209e-09 / wavelength
        calibrant.wavelength = wavelength
        
        print("Setting geometry space...")
        print(f"Search space: {self.param_space}")
        ga_history = {}
        means = [np.mean(bounds[param]) for param in self.param_space]
        scales = [np.max(bounds[param]) - np.min(bounds[param]) for param in self.param_space]
        X_norm = np.random.uniform(low=-1, high=1, size=(n_samples, len(self.param_space)))
        X_samples = X_norm.copy()
        for param in self.param_order:
            idx = self.param_order.index(param)
            if param in self.fix:
                X_samples = np.insert(X_samples, idx, self.default_value[idx], axis=1)
            else:
                X_samples[:, idx] = means[idx] + scales[idx] * X_norm[:, idx]

        scores = np.zeros(n_samples)
        for i in range(n_samples):
            print(f"Initializing sample {i+1}...")
            dist, poni1, poni2, rot1, rot2, rot3 = X_samples[i]
            geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
            score = sg.geometry_refinement.refine3(fix=fix)
            scores[i] = score
            ga_history[f'init_sample_{i+1}'] = {'param':X_samples[i], 'optim': sg.geometry_refinement.param, 'score': score}
        
        for k in range(num_iterations):
            elite_idx = np.argsort(scores)[:m_elite]
            new_gen = []
            for _ in range(n_samples - 1):
                elite_sample = X_norm[np.random.choice(elite_idx)]
                perturbed_sample = elite_sample + np.random.normal(0, sigma, size=len(self.param_order))
                new_gen.append(perturbed_sample)

            best_elite_sample = X_norm[elite_idx[0]]
            new_gen.append(best_elite_sample)
            X_norm = np.array(new_gen)

            X_samples = X_norm.copy()
            for param in self.param_order:
                idx = self.param_order.index(param)
                if param in self.fix:
                    X_samples = np.insert(X_samples, idx, self.default_value[idx], axis=1)
                else:
                    X_samples[:, idx] = means[idx] + scales[idx] * X_norm[:, idx]

            scores = np.zeros(n_samples)
            for i in range(n_samples):
                dist, poni1, poni2, rot1, rot2, rot3 = X_samples[i]
                geom_initial = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=wavelength)
                sg = SingleGeometry("extract_cp", powder_img, calibrant=calibrant, detector=self.detector, geometry=geom_initial)
                sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=8*photon_energy)
                score = sg.geometry_refinement.refine3(fix=fix)
                scores[i] = score
            best_idx = np.argmin(scores)
            ga_history[f'generation_{k+1}_best_sample'] = {'param':X_samples[best_idx], 'score': scores[best_idx]}
        return ga_history, X_samples[best_idx], scores[best_idx]