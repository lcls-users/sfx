from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import warnings

from btx.processing.btx_types import MeasureEMDInput, MeasureEMDOutput

class MeasureEMD:
    """Calculate Earth Mover's Distance between pixel histograms and background."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EMD calculation task.
        
        Args:
            config: Dictionary containing:
                - setup.background_roi_coords: [x1, x2, y1, y2]
                - calculate_emd.num_permutations: Number of bootstrap samples
        """
        self.config = config
        
        # Set defaults
        if 'calculate_emd' not in self.config:
            self.config['calculate_emd'] = {}
        if 'num_permutations' not in self.config['calculate_emd']:
            self.config['calculate_emd']['num_permutations'] = 1000
            
    def _validate_background_roi(self, histograms: np.ndarray) -> None:
        """Validate background ROI against data dimensions."""
        x1, x2, y1, y2 = self.config['setup']['background_roi_coords']
        
        if not (0 <= x1 < x2 <= histograms.shape[1] and
                0 <= y1 < y2 <= histograms.shape[2]):
            raise ValueError(
                f"Background ROI {[x1, x2, y1, y2]} invalid for histograms "
                f"of shape {histograms.shape}"
            )
            
        # Warn if ROI might be too small
        roi_size = (x2 - x1) * (y2 - y1)
        if roi_size < 100:
            warnings.warn(
                f"Background ROI contains only {roi_size} pixels, "
                "which may lead to unstable statistics",
                RuntimeWarning
            )
            
    def _get_average_roi_histogram(
        self, 
        histograms: np.ndarray,
        x1: int,
        x2: int,
        y1: int,
        y2: int
    ) -> np.ndarray:
        """Calculate average histogram for ROI."""
        roi_histograms = histograms[:, x1:x2, y1:y2]
        return np.mean(roi_histograms, axis=(1, 2))
        
    def _calculate_emd_values(
        self,
        histograms: np.ndarray,
        reference_histogram: np.ndarray
    ) -> np.ndarray:
        """Calculate EMD between each pixel's histogram and reference."""
        shape = histograms.shape
        emd_values = np.zeros((shape[1], shape[2]))
        
        for i in range(shape[1]):
            for j in range(shape[2]):
                emd_values[i, j] = wasserstein_distance(
                    histograms[:, i, j],
                    reference_histogram
                )
        
        return emd_values
        
    def _generate_null_distribution(
        self,
        histograms: np.ndarray,
        avg_histogram: np.ndarray
    ) -> np.ndarray:
        """Generate null distribution via bootstrapping."""
        x1, x2, y1, y2 = self.config['setup']['background_roi_coords']
        roi_histograms = histograms[:, x1:x2, y1:y2]
        
        num_bins = roi_histograms.shape[0]
        num_x = x2 - x1
        num_y = y2 - y1
        num_permutations = self.config['calculate_emd']['num_permutations']
        
        null_emd_values = []
        for _ in range(num_permutations):
            # Randomly sample a pixel from background ROI
            x_idx = np.random.randint(0, num_x)
            y_idx = np.random.randint(0, num_y)
            
            sample_histogram = roi_histograms[:, x_idx, y_idx]
            null_emd_value = wasserstein_distance(
                sample_histogram,
                avg_histogram
            )
            null_emd_values.append(null_emd_value)
        
        return np.array(null_emd_values)

    def run(self, input_data: MeasureEMDInput) -> MeasureEMDOutput:
        """Run EMD calculation.
        
        Args:
            input_data: MeasureEMDInput containing histograms
            
        Returns:
            MeasureEMDOutput containing EMD values and null distribution
            
        Raises:
            ValueError: If background ROI is invalid or empty
        """
        histograms = input_data.histogram_output.histograms
        
        # Validate background ROI
        self._validate_background_roi(histograms)
        
        # Get ROI coordinates
        x1, x2, y1, y2 = self.config['setup']['background_roi_coords']
        
        # Calculate average background histogram
        avg_histogram = self._get_average_roi_histogram(
            histograms, x1, x2, y1, y2
        )
        
        # Validate background is not empty
        if np.all(avg_histogram < 1e-8):
            raise ValueError("Background ROI contains no data")
        
        # Calculate EMD values
        emd_values = self._calculate_emd_values(
            histograms,
            avg_histogram
        )
        
        # Generate null distribution
        null_distribution = self._generate_null_distribution(
            histograms,
            avg_histogram
        )
        
        return MeasureEMDOutput(
            emd_values=emd_values,
            null_distribution=null_distribution,
            avg_histogram=avg_histogram,
            avg_hist_edges=input_data.histogram_output.bin_edges
        )
        
    def plot_diagnostics(self, output: MeasureEMDOutput, save_dir: Path) -> None:
        """Generate diagnostic plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. EMD value spatial distribution
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(output.emd_values, cmap='viridis')
        ax1.set_title('EMD Values')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Background ROI overlay on EMD map
        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(output.emd_values, cmap='viridis')
        x1, x2, y1, y2 = self.config['setup']['background_roi_coords']
        rect = plt.Rectangle(
            (y1, x1), y2-y1, x2-x1,
            fill=False, color='red', linewidth=2
        )
        ax2.add_patch(rect)
        ax2.set_title('Background ROI Location')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Average background histogram
        ax3 = fig.add_subplot(223)
        ax3.semilogy(
            output.avg_hist_edges, # removed last elt truncation
            output.avg_histogram,
            'b-',
            label='Background'
        )
        ax3.set_title('Average Background Histogram')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Counts')
        ax3.grid(True)
        
        # 4. Null distribution
        ax4 = fig.add_subplot(224)
        ax4.hist(
            output.null_distribution,
            bins=50,
            density=True,
            alpha=0.5,
            label='Null'
        )
        ax4.hist(
            output.emd_values.ravel(),
            bins=100,
            density=True,
            alpha=0.5,
            label='Data'
        )
        ax4.axvline(
            np.median(output.null_distribution),
            color='r',
            linestyle='--',
            label='Null Median'
        )
        ax4.set_title('EMD Value Distribution')
        ax4.set_xlabel('EMD Value')
        ax4.set_ylabel('Density')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'measure_emd_diagnostics.png')
        plt.close()
