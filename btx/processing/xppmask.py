from analysis_tasks import calculate_p_values, generate_signal_mask, generate_background_mask
from histogram_analysis import calculate_histograms
from histogram_analysis import get_average_roi_histogram, wasserstein_distance
from pvalues import calculate_emd_values
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

logger = logging.getLogger(__name__)

import os

def save_array_to_png(array, output_path):
    # Create the "images" subdirectory if it doesn't exist
    images_dir = os.path.join(os.path.dirname(output_path), "images")
    os.makedirs(images_dir, exist_ok=True)

    # Construct the PNG file path in the "images" subdirectory
    png_filename = os.path.basename(output_path).replace(".npy", ".png")
    png_path = os.path.join(images_dir, png_filename)

    plt.figure(figsize=(10, 8))
    plt.imshow(array, cmap='viridis')
    plt.colorbar(label='Value')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

class MakeHistogram:
    def __init__(self, config):
        required_keys = ['setup', 'make_histogram']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.hdf5_path = config['make_histogram']['input_file']
        self.dataset_path = config['make_histogram']['dataset']
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['make_histogram']['output_dir'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            with h5py.File(self.hdf5_path, 'r') as hdf5_file:
                data = hdf5_file[self.dataset_path][:]
        except (OSError, KeyError) as e:
            self.logger.error(f"Error loading data from {self.dataset_path}: {e}")
            raise

        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {data.ndim}D")

        self.data = data

    def generate_histograms(self):
        histograms = calculate_histograms(self.data)
        self.histograms = histograms

    def summarize(self):
        summary_str = f"Histogram summary for {self.exp} run {self.run}:\n"
        summary_str += f" Shape: {self.histograms.shape}\n"
        summary_str += f" Min: {self.histograms.min():.2f}, Max: {self.histograms.max():.2f}\n"
        summary_str += f" Mean: {self.histograms.mean():.2f}\n"

        self.summary_str = summary_str
        summary_path = os.path.join(self.output_dir, "histogram_summary.txt")
        with open(summary_path, 'w') as summary_file:
            summary_file.write(summary_str)
        self.summary_str = summary_str

    def save_histograms(self):
        histograms_path = os.path.join(self.output_dir, "histograms.npy")
        np.save(histograms_path, self.histograms)

    def report(self, elog_url):
        summary = self.summary_str
        requests.post(elog_url, json=[
            {"key": "Experiment", "value": self.exp},
            {"key": "Run", "value": self.run},
            {"key": "Histogram shape", "value": str(self.histograms.shape)},
            {"key": "Min value", "value": f"{self.histograms.min():.2f}"},
            {"key": "Max value", "value": f"{self.histograms.max():.2f}"},
            {"key": "Mean", "value": f"{self.histograms.mean():.2f}"},
            {"key": "Summary", "value": summary}
        ])

    @property
    def histogram_summary(self):
        return {
            "Experiment": self.exp,
            "Run": self.run,
            "Histogram shape": str(self.histograms.shape),
            "Min value": f"{self.histograms.min():.2f}",
            "Max value": f"{self.histograms.max():.2f}",
            "Mean": f"{self.histograms.mean():.2f}",
        }

class MeasureEMD:
    def __init__(self, config):
        required_keys = ['calculate_emd', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.histograms_path = config['calculate_emd']['histograms_path']
        self.roi_coords = config['calculate_emd']['roi_coords']
        self.num_permutations = config['calculate_emd'].get('num_permutations', 1000)
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['calculate_emd']['output_path'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_histograms(self):
        if not os.path.exists(self.histograms_path):
            raise FileNotFoundError(f"Histogram file {self.histograms_path} does not exist")

        self.histograms = np.load(self.histograms_path)

        if self.histograms.ndim != 3:
            raise ValueError(f"Expected 3D histograms array, got {self.histograms.ndim}D")

        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        if not (0 <= roi_x_start < roi_x_end <= self.histograms.shape[1] and
                0 <= roi_y_start < roi_y_end <= self.histograms.shape[2]):
            raise ValueError(f"Invalid ROI coordinates for histogram of shape {self.histograms.shape}")

    def save_emd_values(self):
        output_file_npy = os.path.join(self.output_dir, "emd_values.npy")
        output_file_png = os.path.join(self.output_dir, "emd_values.png")
        np.save(output_file_npy, self.emd_values)
        save_array_to_png(self.emd_values, output_file_png)

    def save_null_distribution(self):
        output_file = os.path.join(self.output_dir, "emd_null_dist.npy")
        np.save(output_file, self.null_distribution)

    def calculate_emd(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        if not (0 <= roi_x_start < roi_x_end <= self.histograms.shape[1] and
                0 <= roi_y_start < roi_y_end <= self.histograms.shape[2]):
            raise ValueError(f"Invalid ROI coordinates for histogram of shape {self.histograms.shape}")

        self.avg_hist = get_average_roi_histogram(self.histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
        self.emd_values = calculate_emd_values(self.histograms, self.avg_hist)
        self.generate_null_distribution()

    def generate_null_distribution(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        roi_histograms = self.histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end]

        num_bins = roi_histograms.shape[0]
        num_x_indices = roi_x_end - roi_x_start
        num_y_indices = roi_y_end - roi_y_start

        null_emd_values = []

        for _ in range(self.num_permutations):
            random_x_indices = np.random.choice(range(num_x_indices), size=num_bins)
            random_y_indices = np.random.choice(range(num_y_indices), size=num_bins)

            bootstrap_sample_histogram = roi_histograms[np.arange(num_bins), random_x_indices, random_y_indices]
            null_emd_value = wasserstein_distance(bootstrap_sample_histogram, self.avg_hist)
            null_emd_values.append(null_emd_value)

        self.null_distribution = np.array(null_emd_values)

    def summarize(self):
        summary = (
            f"EMD calculation for {self.exp} run {self.run}:\n"
            f" Histogram shape: {self.histograms.shape}\n"
            f" ROI coordinates: {self.roi_coords}\n"
            f" EMD values shape: {self.emd_values.shape}\n"
            f" EMD min: {np.min(self.emd_values):.3f}, max: {np.max(self.emd_values):.3f}\n"
            f" Null distribution shape: {self.null_distribution.shape}\n"
            f" Null distribution min: {np.min(self.null_distribution):.3f}, max: {np.max(self.null_distribution):.3f}\n"
        )

        with open(f"{self.output_dir}_summary.txt", 'w') as f:
            f.write(summary)

        self.summary = summary

    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)

class CalculatePValues:
    def __init__(self, config):
        required_keys = ['calculate_p_values', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.emd_path = config['calculate_p_values']['emd_path']
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['calculate_p_values']['output_path'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

        if 'null_dist_path' not in config['calculate_p_values']:
            raise ValueError("Missing required configuration key: 'null_dist_path'")
        self.null_dist_path = config['calculate_p_values']['null_dist_path']

    def load_data(self):
        self.emd_values = np.load(self.emd_path)
        if self.emd_values.ndim != 2:
            raise ValueError(f"Expected 2D EMD values array, got {self.emd_values.ndim}D")

        self.null_distribution = np.load(self.null_dist_path)

    def calculate_p_values(self):
        self.p_values = calculate_p_values(self.emd_values, self.null_distribution)

    def summarize(self):
        summary = (
            f"P-value calculation for {self.exp} run {self.run}:\n"
            f" EMD values shape: {self.emd_values.shape}\n"
            f" Null distribution length: {len(self.null_distribution)}\n"
            f" P-values shape: {self.p_values.shape}\n"
        )

        with open(f"{self.output_dir}_summary.txt", 'w') as f:
            f.write(summary)

        self.summary = summary

    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)

    def save_p_values(self, p_values_path=None):
        if p_values_path is None:
            p_values_path = os.path.join(self.output_dir, "p_values.npy")

        p_values_png_path = os.path.join(self.output_dir, "p_values.png")
        assert os.path.isdir(self.output_dir), f"Output directory {self.output_dir} does not exist"

        np.save(p_values_path, self.p_values)
        save_array_to_png(self.p_values, p_values_png_path)
        assert os.path.isfile(p_values_path), f"P-values file {p_values_path} was not saved correctly"

class BuildPumpProbeMasks:
    def __init__(self, config):
        required_keys = ['generate_masks', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.histograms_path = config['generate_masks']['histograms_path']
        self.p_values_path = config['generate_masks']['p_values_path']
        self.roi_coords = config['generate_masks']['roi_coords']
        self.threshold = config['generate_masks']['threshold']
        self.bg_mask_mult = config['generate_masks']['bg_mask_mult']
        self.bg_mask_thickness = config['generate_masks']['bg_mask_thickness']
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['generate_masks']['output_dir'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_histograms(self):
        if not os.path.exists(self.histograms_path):
            raise FileNotFoundError(f"Histogram file {self.histograms_path} does not exist")

        self.histograms = np.load(self.histograms_path)

        if self.histograms.ndim != 3:
            raise ValueError(f"Expected 3D histograms array, got {self.histograms.ndim}D")

    def load_p_values(self):
        assert os.path.exists(self.p_values_path), f"P-values file {self.p_values_path} does not exist"
        assert os.path.isfile(self.p_values_path), f"P-values path {self.p_values_path} is not a file"

        self.p_values = np.load(self.p_values_path)
        assert self.p_values.ndim == 2, f"Expected 2D p-values array, got {self.p_values.ndim}D"

    def generate_masks(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        self.signal_mask = generate_signal_mask(
            self.p_values, self.threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end, self.histograms
        )

        self.bg_mask = generate_background_mask(self.signal_mask, self.bg_mask_mult, self.bg_mask_thickness)

    def summarize(self):
        summary = (f"Mask generation for {self.exp} run {self.run}:\n"
                   f" P-values shape: {self.p_values.shape}\n"
                   f" Threshold: {self.threshold}\n"
                   f" Signal mask shape: {self.signal_mask.shape}\n"
                   f" Signal mask count: {np.sum(self.signal_mask)}\n"
                   f" Background mask shape: {self.bg_mask.shape}\n"
                   f" Background mask count: {np.sum(self.bg_mask)}\n")
        with open(f"{self.output_dir}_summary.txt", 'w') as f:
            f.write(summary)
        self.summary = summary

    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)

    def save_masks(self):
        signal_mask_path = os.path.join(self.output_dir, "signal_mask.npy")
        bg_mask_path = os.path.join(self.output_dir, "bg_mask.npy")
        np.save(signal_mask_path, self.signal_mask)
        np.save(bg_mask_path, self.bg_mask)
        save_array_to_png(self.signal_mask, os.path.join(self.output_dir, "signal_mask.png"))
        save_array_to_png(self.bg_mask, os.path.join(self.output_dir, "bg_mask.png"))


import json
from pathlib import Path
import numpy as np
from pump_probe import load_data, calculate_signals_and_errors, calculate_pump_probe_signal, plot_pump_probe_data

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

class PumpProbeAnalysis:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['pump_probe_analysis']['output_dir'])

    def run(self):
        # Step 1: Load data
        data = load_data(self.config)

        # Step 2: Calculate signals and errors
        signals_and_errors = calculate_signals_and_errors(data, self.config)

        # Step 3: Calculate pump-probe signal
        pump_probe_results = calculate_pump_probe_signal(signals_and_errors, self.config)

        # Step 4: Plot pump-probe data
        plot_pump_probe_data(pump_probe_results, self.config)

        # Step 5: Group images into stacks by delay and laser condition
        stacks_on, stacks_off = self.group_images_into_stacks(data)

        # Step 6: Generate pump-probe curves
        pump_probe_curves = self.generate_pump_probe_curves(stacks_on, stacks_off)

        # Step 7: Save pump-probe curves
        self.save_pump_probe_curves(pump_probe_curves)

    def group_images_into_stacks(self, data):
        binned_delays = data['binned_delays']
        imgs = data['imgs']
        laser_on_mask = data['laser_on_mask']
        laser_off_mask = data['laser_off_mask']

        stacks_on = self.extract_stacks_by_delay(binned_delays[laser_on_mask], imgs[laser_on_mask])
        stacks_off = self.extract_stacks_by_delay(binned_delays[laser_off_mask], imgs[laser_off_mask])

        return stacks_on, stacks_off

    def extract_stacks_by_delay(self, binned_delays, img_array):
        unique_binned_delays = np.unique(binned_delays)
        stacks = {}

        for delay in unique_binned_delays:
            mask = (binned_delays == delay)
            stack = img_array[mask]
            stacks[delay] = stack

        return stacks

    def generate_pump_probe_curves(self, stacks_on, stacks_off):
        delays = sorted(set(stacks_on.keys()) | set(stacks_off.keys()))
        pump_probe_curves = {'delay': delays, 'laser_on': [], 'laser_off': []}

        for delay in delays:
            if delay in stacks_on and delay in stacks_off:
                laser_on_data = stacks_on[delay]
                laser_off_data = stacks_off[delay]

                laser_on_signal = np.mean(laser_on_data)
                laser_off_signal = np.mean(laser_off_data)

                pump_probe_curves['laser_on'].append(laser_on_signal)
                pump_probe_curves['laser_off'].append(laser_off_signal)
            else:
                print(f"Warning: Missing data for delay {delay}")

        return pump_probe_curves

    def save_pump_probe_curves(self, pump_probe_curves):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / 'pump_probe_curves.json'
        with open(output_file, 'w') as f:
            json.dump(pump_probe_curves, f, indent=2)

    def summarize(self):
        # TODO: Implement summarize method
        pass

    def report(self):
        # TODO: Implement report method
        pass
