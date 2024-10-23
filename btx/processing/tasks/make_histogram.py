from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from btx.processing.btx_types import MakeHistogramInput, MakeHistogramOutput

class MakeHistogram:
    """Generate histograms from XPP data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize histogram generation task.
        
        Args:
            config: Dictionary containing:
                - make_histogram.bin_boundaries: Array of bin boundaries
                - make_histogram.hist_start_bin: Index of first bin to include
        """
        self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'make_histogram' not in self.config:
            raise ValueError("Missing 'make_histogram' section in config")
            
        hist_config = self.config['make_histogram']
        
        # Set defaults if not provided
        if 'bin_boundaries' not in hist_config:
            hist_config['bin_boundaries'] = np.arange(5, 30, 0.2)
        if 'hist_start_bin' not in hist_config:
            hist_config['hist_start_bin'] = 1
        
        bin_boundaries = hist_config['bin_boundaries']
        if not isinstance(bin_boundaries, (list, tuple, np.ndarray)):
            raise ValueError("bin_boundaries must be array-like")
        if len(bin_boundaries) < 2:
            raise ValueError("bin_boundaries must have at least 2 values")
        
        hist_start_bin = hist_config['hist_start_bin']
        if not isinstance(hist_start_bin, (int, np.integer)):
            raise ValueError("hist_start_bin must be an integer")
        if hist_start_bin < 0 or hist_start_bin >= len(bin_boundaries) - 1:
            raise ValueError("hist_start_bin must be between 0 and len(bin_boundaries)-2")

    def _calculate_histograms(
        self,
        data: np.ndarray,
        bin_boundaries: np.ndarray,
        hist_start_bin: int
    ) -> np.ndarray:
        """Calculate histograms for each pixel.
        
        Args:
            data: 3D array (frames, rows, cols)
            bin_boundaries: Array of histogram bin boundaries
            hist_start_bin: Index of first bin to include
            
        Returns:
            3D array of histograms (bins, rows, cols)
        """
        bins = len(bin_boundaries) - 1
        rows, cols = data.shape[1], data.shape[2]
        hist_shape = (bins, rows, cols)
        
        # Reshape the data for easier computation
        reshaped_data = data.reshape(-1, rows * cols)
        
        # Perform digitization
        bin_indices = np.digitize(reshaped_data, bin_boundaries)
        
        # Initialize histograms
        histograms = np.zeros(hist_shape, dtype=np.float64)
        
        # Populate histograms using bincount
        for i in range(rows * cols):
            valid_indices = bin_indices[:, i] < bins
            histograms[:, i // cols, i % cols] = np.bincount(
                bin_indices[:, i][valid_indices],
                minlength=bins
            )
            # Add counts beyond max to first bin
            histograms[hist_start_bin, i // cols, i % cols] += np.sum(
                reshaped_data[:, i] > bin_boundaries[-1]
            )
        
        # Add small constant
        histograms += 1e-9
        
        # Return histograms starting from hist_start_bin
        return histograms[hist_start_bin:, :, :]

    def run(self, input_data: MakeHistogramInput) -> MakeHistogramOutput:
        """Run histogram generation."""
        hist_config = self.config['make_histogram']
        bin_boundaries = np.array(hist_config['bin_boundaries'])
        hist_start_bin = hist_config['hist_start_bin']
        
        data = input_data.load_data_output.data
        
        histograms = self._calculate_histograms(
            data,
            bin_boundaries,
            hist_start_bin
        )
        
        # Calculate bin edges and centers correctly
        bin_edges = bin_boundaries[hist_start_bin:-1]  # Exclude the last edge
        bin_centers = (bin_boundaries[hist_start_bin:-1] + bin_boundaries[hist_start_bin+1:]) / 2
        
        return MakeHistogramOutput(
            histograms=histograms,
            bin_edges=bin_edges,
            bin_centers=bin_centers
        )

    def plot_diagnostics(self, output: MakeHistogramOutput, save_dir: Path) -> None:
        """Generate diagnostic plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Mean histogram across all pixels (log scale)
        ax1 = fig.add_subplot(221)
        mean_hist = np.mean(output.histograms, axis=(1, 2))
        # Ensure x and y have same length
        ax1.semilogy(output.bin_centers, mean_hist, 'b-')
        ax1.set_title('Mean Histogram Across Pixels (Log Scale)')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Counts')
        ax1.grid(True)
        
        # 2. Histogram variation across pixels
        ax2 = fig.add_subplot(222)
        p25 = np.percentile(output.histograms, 25, axis=(1,2))
        p50 = np.percentile(output.histograms, 50, axis=(1,2))
        p75 = np.percentile(output.histograms, 75, axis=(1,2))
        # Ensure all arrays have same length before plotting
        ax2.fill_between(output.bin_centers, p25, p75, alpha=0.3, label='25-75 percentile')
        ax2.plot(output.bin_centers, p50, 'r-', label='Median')
        ax2.set_title('Histogram Variation Across Pixels')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Counts')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')  # Add log scale here too
        
        # 3. 2D map of total counts
        ax3 = fig.add_subplot(223)
        total_counts = np.sum(output.histograms, axis=0)
        im3 = ax3.imshow(total_counts, cmap='viridis')
        ax3.set_title('Total Counts Map')
        plt.colorbar(im3, ax=ax3, label='Total Counts')
        
        # 4. 2D map of peak positions
        ax4 = fig.add_subplot(224)
        peak_positions = output.bin_centers[np.argmax(output.histograms, axis=0)]
        im4 = ax4.imshow(peak_positions, cmap='viridis')
        ax4.set_title('Peak Position Map')
        plt.colorbar(im4, ax=ax4, label='Peak Position')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'make_histogram_diagnostics.png')
        plt.close()

        # Additional diagnostic: Histogram stack plot
        fig, ax = plt.subplots(figsize=(10, 6))
        center_row = output.histograms.shape[1] // 2
        center_col = output.histograms.shape[2] // 2
        radius = 2
        
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                row = center_row + i
                col = center_col + j
                if 0 <= row < output.histograms.shape[1] and 0 <= col < output.histograms.shape[2]:
                    label = f'Pixel ({row},{col})'
                    ax.semilogy(output.bin_centers, output.histograms[:, row, col], 
                              alpha=0.5, label=label)
        
        ax.set_title('Histograms for Central Pixels')
        ax.set_xlabel('Value')
        ax.set_ylabel('Counts (Log Scale)')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'make_histogram_central_pixels.png')
        plt.close()

        # Add a new diagnostic: Cumulative counts
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_hist = np.mean(output.histograms, axis=(1, 2))
        cumsum = np.cumsum(mean_hist)
        ax.plot(output.bin_centers, cumsum / cumsum[-1], 'b-')
        ax.set_title('Cumulative Distribution (Mean across pixels)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Cumulative Fraction')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / 'make_histogram_cumulative.png')
        plt.close()
