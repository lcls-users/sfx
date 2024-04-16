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
        self.det_type = config['setup']['det_type']
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

        if self.det_type == 'XXXX':
            data = data.reshape((-1, 16))
        elif self.det_type == 'YYYY':
            data = data.astype(np.float32)

        self.data = data

    def generate_histograms(self):
        histograms = calculate_histograms(self.data)
        self.histograms = histograms

    def summarize(self):
        summary_str = f"Histogram summary for {self.exp} run {self.run}:\n"
        summary_str += f"  Shape: {self.histograms.shape}\n"
        summary_str += f"  Min: {self.histograms.min():.2f}, Max: {self.histograms.max():.2f}\n" 
        summary_str += f"  Mean: {self.histograms.mean():.2f}\n"
        
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
        required_keys = ['emd', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        self.histograms_path = config['emd']['histograms_path']
        self.roi_coords = config['emd']['roi_coords']
        self.num_permutations = config['emd'].get('num_permutations', 1000)

        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['emd']['output_dir'], f"{self.exp}_{self.run}")
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
        summary = (f"EMD calculation for {self.exp} run {self.run}:\n"
                   f"  Histogram shape: {self.histograms.shape}\n"
                   f"  ROI coordinates: {self.roi_coords}\n"
                   f"  EMD values shape: {self.emd_values.shape}\n"
                   f"  EMD min: {np.min(self.emd_values):.3f}, max: {np.max(self.emd_values):.3f}\n"
                   f"  Null distribution shape: {self.null_distribution.shape}\n"
                   f"  Null distribution min: {np.min(self.null_distribution):.3f}, max: {np.max(self.null_distribution):.3f}\n")
        
        with open(f"{self.output_dir}_summary.txt", 'w') as f:
            f.write(summary)
            
        self.summary = summary
            
    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)


class CalculatePValues:
    def __init__(self, config, null_distribution):
        required_keys = ['pvalues', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        self.emd_path = config['pvalues']['emd_path'] 
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['pvalues']['output_path'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.null_distribution = null_distribution

    def load_data(self):
        self.emd_values = np.load(self.emd_path)
        if self.emd_values.ndim != 2:
            raise ValueError(f"Expected 2D EMD values array, got {self.emd_values.ndim}D")
            
    def calculate_p_values(self):
        self.p_values = calculate_p_values(self.emd_values, self.null_distribution)
        
    def summarize(self):
        summary = (f"P-value calculation for {self.exp} run {self.run}:\n"
                   f"  EMD values shape: {self.emd_values.shape}\n"  
                   f"  Null distribution length: {len(self.null_distribution)}\n"
                   f"  P-values shape: {self.p_values.shape}\n")
        
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
        required_keys = ['masks', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        self.histograms_path = config['masks']['histograms_path']
        self.p_values_path = config['masks']['p_values_path']
        self.roi_coords = config['masks']['roi_coords']
        self.threshold = config['masks']['threshold']
        self.bg_mask_mult = config['masks']['bg_mask_mult']
        self.bg_mask_thickness = config['masks']['bg_mask_thickness']
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['masks']['output_dir'], f"{self.exp}_{self.run}")
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

