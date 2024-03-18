from btx.diagnostics.run import RunDiagnostics
from btx.diagnostics.ag_behenate import AgBehenate
from btx.interfaces.ipsana import assemble_image_stack_batch
from btx.misc.radial import radial_profile, pix2q, q2pix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from btx.misc.shortcuts import TaskTimer
import scipy as scp
from scipy.stats import norm
import numpy as np
import yaml
import os
from btx.misc.shortcuts import fetch_latest
import argparse


def upper_confidence_bound(X, gp_model, beta):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        return ucb


def bay_opt_geom_subroutine(job_index,
                            root_dir,
                            outdir,
                            run,
                            exp,
                            det_type,
                            length_fraction,
                            resolution,
                            n_samples,
                            num_iterations
                            ):

    taskdir = os.path.join(root_dir, 'geom')
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    powder_file = os.path.join(root_dir, f"powder/r{run:04}_max.npy")
    powder_img = np.load(powder_file)

    diagnostics = RunDiagnostics(exp=exp,
                                run=run,
                                det_type=det_type)
    pixel_size = diagnostics.psi.get_pixel_size()
    wavelength = diagnostics.psi.get_wavelength()

    mask_file = fetch_latest(fnames=os.path.join(root_dir, 'mask', 'r*.npy'), run=run)
    mask_img = np.load(mask_file)
    if diagnostics.psi.det_type != 'Rayonix':
        mask_img = assemble_image_stack_batch(mask_img, diagnostics.pixel_index_map)

    ag_behenate = AgBehenate(powder=powder_img,
                            mask=mask_img,
                            pixel_size=pixel_size,
                            wavelength=wavelength)
    
    height, width = powder_img.shape[:2]
    # Initial guess for the distance
    distance_i = diagnostics.psi.estimate_distance()

    x_MIN = width//2 - width//length_fraction
    x_MAX = width//2 + width//length_fraction
    y_MIN = height//2 - height//length_fraction
    y_MAX = height//2 + height//length_fraction
    
    bo_history = {}

    distances = []
    
    min_max_scaler = MinMaxScaler()
    normalization_factor = 0.3

    cx_mesh, cy_mesh = np.meshgrid(np.arange(x_MIN, x_MAX + resolution, resolution), np.arange(y_MIN, y_MAX + resolution, resolution))
    input_range = np.hstack((cx_mesh.reshape(-1, 1), cy_mesh.reshape(-1, 1)))
    input_range = min_max_scaler.fit_transform(input_range)

    idx_initial = np.random.choice(input_range.shape[0], n_samples).reshape(-1, 1)
    sample_inputs = input_range[idx_initial].reshape(-1, 2)

    sample_y = np.zeros((n_samples, 1))
    for i in np.arange(n_samples):
        cx, cy = float(sample_inputs[i,0]), float(sample_inputs[i,1])
        cx_pixels, cy_pixels = x_MIN + cx*(x_MAX-x_MIN), y_MIN + cy*(y_MAX-y_MIN)

        iprofile = radial_profile(data=powder_img, center=(cx_pixels, cy_pixels), mask=mask_img)
        peaks_observed, properties = scp.signal.find_peaks(iprofile, prominence=1, distance=10)
        qprofile = pix2q(np.arange(iprofile.shape[0]), wavelength, distance_i, pixel_size)
        rings, scores = ag_behenate.ideal_rings(qprofile[peaks_observed])
        peaks_predicted = q2pix(rings, wavelength, distance_i, pixel_size)
        opt_distance = ag_behenate.detector_distance(peaks_predicted[0])
        distances.append(opt_distance)

        score = float(np.min(scores))
        sample_y[i,0] = -score
        bo_history[f'init_sample_{i+1}'] = {'cx': cx, 'cy': cy, 'score': score}

    sample_y = sample_y/normalization_factor
    visited_idx = list(idx_initial.flatten())

    kernel = RBF(length_scale=0.3, length_scale_bounds='fixed') \
                * ConstantKernel(constant_value=1.0, constant_value_bounds=(0.5, 1.5)) \
                + WhiteKernel(noise_level=0.001, noise_level_bounds = 'fixed')
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp_model.fit(sample_inputs, sample_y)

    for i in range(num_iterations):
        # 1. Generate the Acquisition Function values using the Gaussian Process Regressor
        beta = 0.6
        af_values = np.zeros((input_range.shape[0], 1))
        for j in range(input_range.shape[0]):
            af_values[j] = upper_confidence_bound(input_range[j].reshape(1, -1), gp_model, beta)
        af_values[visited_idx] = -np.inf
        
        # 2. Select the next set of parameters based on the Acquisition Function
        new_idx = np.argmax(af_values)
        new_input = input_range[new_idx]
        visited_idx.append(new_idx)

        # 3. Compute the score of the new set of parameters 
        cx, cy = float(new_input[0]), float(new_input[1])
        # cx_pixels, cy_pixels = x_MIN + cx*(x_MAX-x_MIN), y_MIN + cy*(y_MAX-y_MIN)
        center_pixels = min_max_scaler.inverse_transform(new_input.reshape(1, -1))
        cx_pixels, cy_pixels = center_pixels[0,0], center_pixels[0,1]

        iprofile = radial_profile(data=powder_img, center=(cx_pixels, cy_pixels), mask=mask_img)
        peaks_observed, properties = scp.signal.find_peaks(iprofile, prominence=1, distance=10)
        qprofile = pix2q(np.arange(iprofile.shape[0]), wavelength, distance_i, pixel_size)
        rings, scores = ag_behenate.ideal_rings(qprofile[peaks_observed])
        peaks_predicted = q2pix(rings, wavelength, distance_i, pixel_size)
        opt_distance = ag_behenate.detector_distance(peaks_predicted[0])
        distances.append(opt_distance)

        score = float(np.min(scores))
        bo_history[f'iteration_{i+1}'] = {'cx': cx, 'cy': cy, 'score': score}

        sample_y = np.vstack((sample_y, -score/normalization_factor))
        sample_inputs = np.vstack((sample_inputs, [[cx, cy]]))

        # 4. Fit the Gaussian Process Regressor
        gp_model.fit(sample_inputs, sample_y)

    best_index = np.argmax(normalization_factor*sample_y)
    best_score = float(normalization_factor*sample_y[best_index][0])
    center_pixels = min_max_scaler.inverse_transform(sample_inputs[best_index].reshape(1, -1))
    cx_min, cy_min = float(center_pixels[0,0]), float(center_pixels[0,1])
    opt_distance = float(distances[best_index])

    # Save the results in a .yaml file
    with open(os.path.join(outdir, f'bay_opt_geom_{length_fraction//2}th_length_{n_samples}_points_{num_iterations}_iter_job_{job_index}_results.yaml'), 'w') as yaml_file:
        result_dict = {"best_score": best_score, "cx_min": cx_min, "cy_min": cy_min, "opt_distance": opt_distance}
        yaml.dump(result_dict, yaml_file, default_flow_style=False, sort_keys=False)

    # Save the bo history in a .yaml file
    with open(os.path.join(outdir, f'bay_opt_geom_{length_fraction//2}th_length_{n_samples}_points_{num_iterations}_iter_job_{job_index}_history.yaml'), 'w') as yaml_file:
        yaml.dump(bo_history, yaml_file, default_flow_style=False, sort_keys=False)



def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_index', help='Job index', required=True, type=int)
    parser.add_argument('--root_dir', help='Root directory',  required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory for cxi files', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('--length_fraction', help='Length fraction', required=True, type=int)
    parser.add_argument('--resolution', help='Resolution', required=True, type=float)
    parser.add_argument('--n_samples', help='Number of initial samples', required=True, type=int)
    parser.add_argument('--num_iterations', help='Number of iterations', required=True, type=int)
    
    return parser.parse_args()



if __name__ == '__main__':
    
    params = parse_input()
    bay_opt_geom_subroutine(job_index=params.job_index,
                            root_dir=params.root_dir,
                            outdir=params.outdir,
                            run=params.run,
                            exp=params.exp,
                            det_type=params.det_type,
                            length_fraction=params.length_fraction,
                            resolution=params.resolution,
                            n_samples=params.n_samples,
                            num_iterations=params.num_iterations)