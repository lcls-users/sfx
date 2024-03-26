from analysis_tasks import calculate_p_values, generate_signal_mask, generate_background_mask
from histogram_analysis import calculate_histograms
from histogram_analysis import get_average_roi_histogram, wasserstein_distance
from pvalues import calculate_emd_values
import h5py
import logging
import numpy as np
import os
import requests
# need to import these functions

logger = logging.getLogger(__name__)

class MakeHistogram:
    """
    Class for generating normalized histograms from HDF5 data.
    """

    def __init__(self, config):
        """
        Initialize a MakeHistogram instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary
        """
        required_keys = ['setup', 'make_histogram']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.hdf5_path = os.path.join(config['setup']['root_dir'], config['make_histogram']['dataset'])
        self.exp = config['setup']['exp'] 
        self.run = config['setup']['run']
        self.det_type = config['setup']['det_type']
        
        self.output_dir = config['make_histogram']['output_dir']

        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Load data from the HDF5 file specified in the configuration.
        """
        try:
            with h5py.File(self.hdf5_path, 'r') as hdf5_file:
                data = hdf5_file[self.hdf5_path][:]
        except (OSError, KeyError) as e:
            self.logger.error(f"Error loading data from {self.hdf5_path}: {e}")
            raise

        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {data.ndim}D")

        self.data = data

    def generate_histograms(self):
        """
        Generate normalized histograms from the loaded data.
        """
        histograms = calculate_histograms(self.data)
        self.histograms = histograms

    def summarize(self):
        """
        Generate a summary string and write it to a file.
        """
        summary_str = f"Histogram summary for {self.exp} run {self.run}:\n"
        summary_str += f"  Shape: {self.histograms.shape}\n"
        summary_str += f"  Min: {self.histograms.min():.2f}, Max: {self.histograms.max():.2f}\n" 
        summary_str += f"  Mean: {self.histograms.mean():.2f}\n"
        
        summary_path = os.path.join(self.output_dir, f"make_histogram_{self.exp}_{self.run}.summary")
        with open(summary_path, 'w') as summary_file:
            summary_file.write(summary_str)
        
        self.summary_str = summary_str

    def save_histograms(self):
        """
        Save the generated histograms to a file.
        """
        histograms_path = os.path.join(self.output_dir, f"histograms_{self.exp}_{self.run}.npy")
        np.save(histograms_path, self.histograms)
        
    def report(self, elog_url):
        """
        Post a summary of the histograms to the specified Elog URL.

        Args:
            elog_url (str): URL to post the summary to
        """
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
        """
        Return a dictionary summarizing the generated histograms.
        """
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
        self.output_path = config['emd']['output_path']
        self.roi_coords = config['emd']['roi_coords']
        self.num_permutations = config['emd'].get('num_permutations', 1000)
        
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        
    def load_histograms(self):
        histograms_path = os.path.join(self.histograms_path, f"histograms_{self.exp}_{self.run}.npy")
        self.histograms = np.load(histograms_path)

        
        if self.histograms.ndim != 3:
            raise ValueError(f"Expected 3D histograms array, got {self.histograms.ndim}D")
            
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords
        if not (0 <= roi_x_start < roi_x_end <= self.histograms.shape[1] and 
                0 <= roi_y_start < roi_y_end <= self.histograms.shape[2]):
            raise ValueError(f"Invalid ROI coordinates for histogram of shape {self.histograms.shape}")
        
    def calculate_emd(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords
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
        
        with open(f"{self.output_path}_summary.txt", 'w') as f:
            f.write(summary)
            
        self.summary = summary
            
    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)
            
    def save_emd_values(self):
        emd_values_path = os.path.join(self.output_path, f"emd_values_{self.exp}_{self.run}.npy")
        np.save(emd_values_path, self.emd_values)
        
    def save_null_distribution(self):
        null_dist_path = os.path.join(self.output_path, f"null_distribution_{self.exp}_{self.run}.npy")
        np.save(null_dist_path, self.null_distribution)


class CalculatePValues:
    def __init__(self, config):
        required_keys = ['pvalues', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        self.emd_path = config['pvalues']['emd_path'] 
        self.null_dist_path = config['pvalues']['null_dist_path']
        self.output_path = config['pvalues']['output_path']
        
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        
    def load_data(self):
        emd_values_path = os.path.join(self.emd_path, f"emd_values_{self.exp}_{self.run}.npy")
        self.emd_values = np.load(emd_values_path)
        if self.emd_values.ndim != 2:
            raise ValueError(f"Expected 2D EMD values array, got {self.emd_values.ndim}D")
            

        null_dist_path = os.path.join(self.null_dist_path, f"null_distribution_{self.exp}_{self.run}.npy")
        self.null_dist = np.load(null_dist_path)
        if self.null_dist.ndim != 1:
            raise ValueError(f"Expected 1D null distribution array, got {self.null_dist.ndim}D")
            
    def calculate_p_values(self):
        self.p_values = calculate_p_values(self.emd_values, self.null_dist)
        
    def summarize(self):
        summary = (f"P-value calculation for {self.exp} run {self.run}:\n"
                   f"  EMD values shape: {self.emd_values.shape}\n"  
                   f"  Null distribution length: {len(self.null_dist)}\n"
                   f"  P-values shape: {self.p_values.shape}\n")
        
        with open(f"{self.output_path}_summary.txt", 'w') as f:
            f.write(summary)
            
        self.summary = summary
        
    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)
            
    def save_p_values(self):
        p_values_path = os.path.join(self.output_path, f"p_values_{self.exp}_{self.run}.npy")
        np.save(p_values_path, self.p_values)
        

class BuildPumpProbeMasks:
    def __init__(self, config):
        required_keys = ['masks', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        self.p_values_path = config['masks']['p_values_path']
        self.output_dir = config['masks']['output_dir']  
        
        self.roi_coords = config['masks']['roi_coords']
        self.threshold = config['masks']['threshold']
        self.bg_mask_mult = config['masks']['bg_mask_mult']
        self.bg_mask_thickness = config['masks']['bg_mask_thickness']
        
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        
    def load_p_values(self):
        p_values_path = os.path.join(self.p_values_path, f"p_values_{self.exp}_{self.run}.npy")
        self.p_values = np.load(p_values_path)

        
        if self.p_values.ndim != 2:
            raise ValueError(f"Expected 2D p-values array, got {self.p_values.ndim}D")

        if not 0 <= self.threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")            

        if self.bg_mask_mult < 0:
            raise ValueError(f"Background mask multiple must be non-negative, got {self.bg_mask_mult}")

        if self.bg_mask_thickness < 0:
            raise ValueError(f"Background mask thickness must be non-negative, got {self.bg_mask_thickness}")

    def generate_masks(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords
        self.signal_mask = generate_signal_mask(
                self.p_values, self.threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end)

        self.bg_mask = generate_background_mask(self.signal_mask, self.bg_mask_mult, self.bg_mask_thickness)

    def summarize(self):
        summary = (f"Mask generation for {self.exp} run {self.run}:\n"
                   f"  P-values shape: {self.p_values.shape}\n"
                   f"  Threshold: {self.threshold}\n"  
                   f"  Signal mask shape: {self.signal_mask.shape}\n"
                   f"  Signal mask count: {np.sum(self.signal_mask)}\n"
                   f"  Background mask shape: {self.bg_mask.shape}\n"
                   f"  Background mask count: {np.sum(self.bg_mask)}\n")

        with open(f"{self.output_dir}_summary.txt", 'w') as f: 
            f.write(summary)

        self.summary = summary

    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary) 

    def save_masks(self):
        signal_mask_path = os.path.join(self.output_dir, f"signal_mask_{self.exp}_{self.run}.npy")
        bg_mask_path = os.path.join(self.output_dir, f"bg_mask_{self.exp}_{self.run}.npy")
        np.save(signal_mask_path, self.signal_mask) 
        np.save(bg_mask_path, self.bg_mask)
