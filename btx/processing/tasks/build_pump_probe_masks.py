from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import warnings

from btx.processing.btx_types import (
    BuildPumpProbeMasksInput,
    BuildPumpProbeMasksOutput,
    SignalMaskStages,
    MakeHistogramOutput,
    CalculatePValuesOutput
)

class BuildPumpProbeMasks:
    """Generate signal and background masks from p-values with ROI-based clustering.
    
    This implementation uses ROI-connected clustering for signal identification,
    with data-aware rectification and proper handling of negative clusters. The
    background mask is generated using binary dilation targeting a specific size
    relative to the signal mask.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mask generation task.
        
        Args:
            config: Dictionary containing:
                - setup.background_roi_coords: [x1, x2, y1, y2]
                - generate_masks.threshold: P-value threshold
                - generate_masks.bg_mask_mult: Background mask multiplier
                - generate_masks.bg_mask_thickness: Background mask thickness
        """
        self.config = config
        
        # Set defaults
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'setup' not in self.config:
            raise ValueError("Missing 'setup' section in config")
            
        if 'background_roi_coords' not in self.config['setup']:
            raise ValueError("Missing background_roi_coords in setup")
            
        roi = self.config['setup']['background_roi_coords']
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("background_roi_coords must be [x1, x2, y1, y2]")
            
        # Validate generate_masks section
        if 'generate_masks' not in self.config:
            raise ValueError("Missing 'generate_masks' section in config")
            
        masks_config = self.config['generate_masks']
        required = ['threshold', 'bg_mask_mult', 'bg_mask_thickness']
        for param in required:
            if param not in masks_config:
                raise ValueError(f"Missing required parameter: {param}")
                
        if not 0 < masks_config['threshold'] < 1:
            raise ValueError("threshold must be between 0 and 1")
            
        if masks_config['bg_mask_mult'] <= 0:
            raise ValueError("bg_mask_mult must be positive")
            
        if masks_config['bg_mask_thickness'] <= 0:
            raise ValueError("bg_mask_thickness must be positive")
        if 'generate_masks' not in self.config:
            self.config['generate_masks'] = {}
        
        masks_config = self.config['generate_masks']
        if 'threshold' not in masks_config:
            masks_config['threshold'] = 0.05
        if 'bg_mask_mult' not in masks_config:
            masks_config['bg_mask_mult'] = 2.0
        if 'bg_mask_thickness' not in masks_config:
            masks_config['bg_mask_thickness'] = 5

    def _rectify_filter_mask(self, mask: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Rectify mask orientation based on data values.
        
        Args:
            mask: Binary mask to rectify
            data: Original histogram data
            
        Returns:
            Rectified binary mask
        """
        imgs_sum = data.sum(axis=0)
        
        if mask.sum() == 0:
            return ~mask
            
        mean_1 = imgs_sum[mask].mean()
        mean_0 = imgs_sum[~mask].mean()
        
        return ~mask if mean_1 < mean_0 else mask

    def _identify_roi_connected_cluster(
        self,
        p_values: np.ndarray,
        threshold: float,
        roi_x_start: int,
        roi_x_end: int,
        roi_y_start: int,
        roi_y_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find cluster connected to ROI.
        
        Args:
            p_values: Array of p-values
            threshold: P-value threshold
            roi_x_start, roi_x_end, roi_y_start, roi_y_end: ROI coordinates
            
        Returns:
            Tuple of (labeled array, ROI cluster mask)
        """
        porous_pixels = p_values > threshold
        labeled_array, _ = ndimage.label(porous_pixels)
        seed_x = (roi_x_start + roi_x_end) // 2
        seed_y = (roi_y_start + roi_y_end) // 2
        roi_cluster_label = labeled_array[seed_x, seed_y]
        return labeled_array, labeled_array == roi_cluster_label

    def _filter_negative_clusters(
        self,
        cluster_array: np.ndarray,
        data: np.ndarray,
        min_size: int = 10
    ) -> np.ndarray:
        """Filter out small negative clusters.
        
        Args:
            cluster_array: Binary array of clusters
            data: Original histogram data
            min_size: Minimum cluster size to keep
            
        Returns:
            Filtered binary mask
        """
        # First rectify the mask orientation
        cluster_array = self._rectify_filter_mask(cluster_array, data)
        
        # Invert array to work with negative clusters
        inverted_array = np.logical_not(cluster_array)
        
        # Label inverted regions
        labeled_array, _ = ndimage.label(inverted_array)
        
        # Count size of each cluster
        cluster_sizes = np.bincount(labeled_array.ravel())
        
        # Find small clusters
        small_clusters = np.where(cluster_sizes < min_size)[0]
        
        # Create mask of small clusters
        small_cluster_mask = np.isin(labeled_array, small_clusters)
        
        # Return original mask with small negative clusters filled
        return np.logical_or(cluster_array, small_cluster_mask)

    def _infill_binary_array(self, data: np.ndarray, array: np.ndarray) -> np.ndarray:
        """Fill holes keeping only largest component.
        
        Args:
            data: Original histogram data
            array: Binary array to infill
            
        Returns:
            Infilled binary mask
        """
        # Rectify mask orientation again
        labeled_array, num_features = ndimage.label(
            self._rectify_filter_mask(array, data)
        )
        
        # Find largest component
        largest_component = 0
        largest_size = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            if size > largest_size:
                largest_size = size
                largest_component = i
                
        return labeled_array == largest_component

    def _create_continuous_buffer(
        self,
        signal_mask: np.ndarray,
        initial_thickness: int = 10,
        num_pixels: Optional[int] = None,
        separator_thickness: int = 5
    ) -> np.ndarray:
        """Create continuous buffer around signal targeting specific size.
        
        Args:
            signal_mask: Binary signal mask
            initial_thickness: Initial dilation thickness
            num_pixels: Target number of pixels in buffer
            separator_thickness: Thickness of separator between signal and buffer
            
        Returns:
            Binary buffer mask
        """
        if num_pixels is not None:
            available_space = np.prod(signal_mask.shape) - np.sum(signal_mask)
            if num_pixels > available_space:
                raise ValueError("Target pixels exceeds available space")
        
        if signal_mask.sum() == 0:
            raise ValueError("Signal mask is empty")
        
        # Create separator gap
        dilated_signal = binary_dilation(signal_mask, iterations=separator_thickness)
        
        # Grow buffer until target size reached
        current_num_pixels = 0
        thickness = 0
        while num_pixels is not None and current_num_pixels < num_pixels:
            thickness += 1
            buffer = binary_dilation(dilated_signal, iterations=thickness) & (~dilated_signal)
            current_num_pixels = np.sum(buffer)
            
        return buffer

    def _create_background_mask(
        self,
        signal_mask: np.ndarray,
        bg_mask_mult: float,
        thickness: int,
        separator_thickness: int = 5
    ) -> np.ndarray:
        """Create background mask targeting specific size relative to signal.
        
        Args:
            signal_mask: Binary signal mask
            bg_mask_mult: Multiple of signal mask size for background
            thickness: Initial thickness for dilation
            separator_thickness: Thickness of separator between signal and background
            
        Returns:
            Binary background mask
        """
        num_pixels_signal = np.sum(signal_mask)
        target_bg_pixels = int(num_pixels_signal * bg_mask_mult)
        
        return self._create_continuous_buffer(
            signal_mask,
            initial_thickness=thickness,
            num_pixels=target_bg_pixels,
            separator_thickness=separator_thickness
        )

    def _validate_inputs(self, input_data: BuildPumpProbeMasksInput) -> None:
        return
        """Validate input data dimensions."""
        p_values = input_data.p_values_output.p_values
        histograms = input_data.histogram_output.histograms
        
        # Check histogram data exists
        if histograms is None:
            raise ValueError("Missing histogram data")
        
        # Check dimensions match
        if p_values.shape != histograms.shape[1:]:
            raise ValueError(
                f"P-value shape {p_values.shape} doesn't match "
                f"histogram shape {histograms.shape[1:]}"
            )
        
        # Validate ROI coordinates
        x1, x2, y1, y2 = self.config['setup']['background_roi_coords']
        if not (0 <= x1 < x2 <= p_values.shape[0] and
                0 <= y1 < y2 <= p_values.shape[1]):
            raise ValueError(
                f"ROI coordinates {[x1, x2, y1, y2]} invalid for "
                f"data shape {p_values.shape}"
            )

    def _validate_masks(
        self,
        signal_mask: np.ndarray,
        background_mask: np.ndarray
    ) -> None:
        """Validate final masks."""
        # Check for overlap
        if np.any(signal_mask & background_mask):
            raise ValueError("Signal and background masks overlap")
        
        # Check sizes
        signal_size = np.sum(signal_mask)
        background_size = np.sum(background_mask)
        total_size = signal_mask.size
        
        if signal_size == 0:
            raise ValueError("Signal mask is empty")
        if background_size == 0:
            raise ValueError("Background mask is empty")
            
        # Check relative sizes
        if background_size < signal_size * self.config['generate_masks']['bg_mask_mult'] * 0.9:
            warnings.warn(
                f"Background mask smaller than expected: {background_size} pixels vs "
                f"target {signal_size * self.config['generate_masks']['bg_mask_mult']}",
                RuntimeWarning
            )

    def process(self, config: Dict[str, Any],
                histogram_output: MakeHistogramOutput,
                p_values_output: CalculatePValuesOutput) -> BuildPumpProbeMasksOutput:
        """Process mask generation directly from inputs.
        
        Args:
            config: Configuration dictionary
            histogram_output: Output from MakeHistogram task
            p_values_output: Output from CalculatePValues task
            
        Returns:
            BuildPumpProbeMasksOutput containing signal and background masks
        """
        input_data = BuildPumpProbeMasksInput(
            config=config,
            histogram_output=histogram_output,
            p_values_output=p_values_output
        )
        return self.run(input_data)

    def run(self, input_data: BuildPumpProbeMasksInput) -> BuildPumpProbeMasksOutput:
        """Run mask generation."""
        # Validate inputs
        self._validate_inputs(input_data)
        
        # Get parameters
        mask_config = self.config['generate_masks']
        threshold = mask_config['threshold']
        bg_mult = mask_config['bg_mask_mult']
        thickness = mask_config['bg_mask_thickness']
        x1, x2, y1, y2 = self.config['setup']['background_roi_coords']
        
        # Get data
        p_values = input_data.p_values_output.p_values
        histograms = input_data.histogram_output.histograms
        
        # 1. Initial ROI cluster identification
        _, roi_cluster = self._identify_roi_connected_cluster(
            p_values, threshold, x1, x2, y1, y2
        )
        
        # 2. Filter negative clusters
        filtered_mask = self._filter_negative_clusters(
            roi_cluster, histograms, min_size=10
        )
        
        # 3. Infill to get final signal mask
        final_signal_mask = self._infill_binary_array(histograms, filtered_mask)
        
        # 4. Generate background mask
        background_mask = self._create_background_mask(
            final_signal_mask,
            bg_mask_mult=bg_mult,
            thickness=thickness,
            separator_thickness=5
        )
        
        # 5. Validate masks
        self._validate_masks(final_signal_mask, background_mask)
        
        # 6. Print statistics
        self._print_mask_statistics(final_signal_mask, background_mask)
        
        # Store intermediate results
        intermediate_masks = SignalMaskStages(
            initial=roi_cluster,
            roi_masked=filtered_mask,
            filtered=filtered_mask,  # Reuse since filtering is different now
            final=final_signal_mask
        )
        
        return BuildPumpProbeMasksOutput(
            signal_mask=final_signal_mask,
            background_mask=background_mask,
            intermediate_masks=intermediate_masks
        )

    def _print_mask_statistics(
        self,
        signal_mask: np.ndarray,
        background_mask: np.ndarray
    ) -> None:
        """Print mask statistics."""
        total_pixels = signal_mask.size
        signal_pixels = np.sum(signal_mask)
        background_pixels = np.sum(background_mask)
        
        print("\nMask Statistics:")
        print(f"Signal mask: {signal_pixels} pixels "
              f"({signal_pixels/total_pixels:.1%} of image)")
        print(f"Background mask: {background_pixels} pixels "
              f"({background_pixels/total_pixels:.1%} of image)")
        
        # Calculate minimum distance between masks
        signal_dist = ndimage.distance_transform_edt(~signal_mask)
        min_distance = np.min(signal_dist[background_mask])
        print(f"Minimum distance between masks: {min_distance:.1f} pixels")

    def plot_diagnostics(
            self,
            output: BuildPumpProbeMasksOutput,
            save_dir: Path
        ) -> None:
            """Generate diagnostic plots.
            
            Args:
                output: Output from mask generation
                save_dir: Directory to save plots
            """
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Plot mask generation stages
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle('Signal Mask Generation Stages')
            
            stages = [
                ('Initial ROI Cluster', output.intermediate_masks.initial),
                ('Filtered Clusters', output.intermediate_masks.roi_masked),
                ('Final Signal Mask', output.intermediate_masks.final),
                ('Background Mask', output.background_mask)
            ]
            
            for ax, (title, mask) in zip(axes.flat, stages):
                ax.imshow(mask, cmap='RdBu')
                ax.set_title(title)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'mask_generation_stages.png')
            plt.close()
            
            # 2. Plot final masks
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(output.signal_mask)
            ax1.set_title('Signal Mask')
            
            ax2.imshow(output.background_mask)
            ax2.set_title('Background Mask')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'final_masks.png')
            plt.close()
            
            # 3. Plot distance transform
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Calculate distance from signal mask
            dist = ndimage.distance_transform_edt(~output.signal_mask)
            
            # Plot distance with background mask contour
            im = ax.imshow(dist, cmap='viridis')
            plt.colorbar(im, ax=ax, label='Distance (pixels)')
            
            # Add background mask contour
            ax.contour(
                output.background_mask,
                colors='r',
                levels=[0.5],
                linewidths=2,
                label='Background Mask'
            )
            
            ax.set_title('Distance from Signal to Background')
            plt.tight_layout()
            plt.savefig(save_dir / 'mask_distance.png')
            plt.close()
