from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import warnings

from btx.processing.types import (
    BuildPumpProbeMasksInput,
    BuildPumpProbeMasksOutput,
    SignalMaskStages
)

class BuildPumpProbeMasks:
    """Generate signal and background masks from p-values."""
    
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
        self._validate_config()
        
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

    def _identify_roi_connected_cluster(
        self,
        mask: np.ndarray,
        x1: int,
        x2: int,
        y1: int,
        y2: int
    ) -> np.ndarray:
        """Identify clusters connected to ROI region."""
        # Label connected components
        labels, num_features = ndimage.label(mask)
        if num_features == 0:
            return np.zeros_like(mask)
            
        # Find unique labels in ROI
        roi_labels = np.unique(labels[x1:x2, y1:y2])
        roi_labels = roi_labels[roi_labels != 0]  # Remove background label
        
        # Create output mask
        result = np.zeros_like(mask)
        for label in roi_labels:
            result |= (labels == label)
            
        return result

    def _filter_clusters(
        self,
        mask: np.ndarray,
        min_size: int = 10
    ) -> np.ndarray:
        """Remove small clusters from mask."""
        # Label connected components
        labels, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask
            
        # Calculate cluster sizes
        sizes = np.bincount(labels.ravel())
        
        # Create mask of clusters to keep
        keep_labels = (sizes > min_size)
        keep_labels[0] = 0  # Don't keep background
        
        return keep_labels[labels]

    def _create_continuous_buffer(
        self,
        signal_mask: np.ndarray,
        thickness: int
    ) -> np.ndarray:
        """Create a continuous buffer around signal regions."""
        buffer = np.zeros_like(signal_mask)
        
        # Label connected regions in signal mask
        signal_labels, num_signals = ndimage.label(signal_mask)
        
        # Process each signal region separately
        for i in range(1, num_signals + 1):
            signal_region = signal_labels == i
            
            # Calculate distance from this signal region
            distances = ndimage.distance_transform_edt(~signal_region)
            
            # Add buffer zone around this region
            buffer |= (distances <= thickness)
        
        return buffer

    def _create_background_mask(
        self,
        signal_with_buffer: np.ndarray,
        bg_mask_mult: float
    ) -> np.ndarray:
        """Create background mask considering signal buffer."""
        # Calculate distances from buffered signal
        distances = ndimage.distance_transform_edt(~signal_with_buffer)
        
        # Background thickness is relative to image size
        bg_thickness = float(signal_with_buffer.shape[0]) * bg_mask_mult
        
        # Create background mask excluding signal buffer
        return (distances <= bg_thickness) & (~signal_with_buffer)

    def run(
        self,
        input_data: BuildPumpProbeMasksInput
    ) -> BuildPumpProbeMasksOutput:
        """Run mask generation."""
        # Validate inputs
        self._validate_inputs(input_data)
        
        # Get parameters
        mask_config = self.config['generate_masks']
        threshold = mask_config['threshold']
        bg_mult = mask_config['bg_mask_mult']
        thickness = mask_config['bg_mask_thickness']
        x1, x2, y1, y2 = self.config['setup']['background_roi_coords']
        
        # 1. Initial thresholding
        initial_mask = input_data.p_values_output.p_values < threshold
        
        # 2. Identify signal clusters (ignoring ROI for now)
        signal_clusters = initial_mask.copy()
        
        # 3. Filter small clusters
        filtered_mask = self._filter_clusters(signal_clusters, min_size=10)
        
        # 4. Infill holes in signal mask
        final_signal_mask = ndimage.binary_fill_holes(filtered_mask)
        
        # 5. Generate background mask
        # First create safety buffer around signal
        continuous_buffer = self._create_continuous_buffer(
            final_signal_mask,
            thickness
        )
        
        # Then create background mask
        background_mask = self._create_background_mask(
            continuous_buffer,
            bg_mult
        )
        
        # 6. Validate masks
        self._validate_masks(final_signal_mask, background_mask)
        
        # 7. Print statistics
        self._print_mask_statistics(final_signal_mask, background_mask)
        
        # Store intermediate results
        intermediate_masks = SignalMaskStages(
            initial=initial_mask,
            roi_masked=signal_clusters,
            filtered=filtered_mask,
            final=final_signal_mask
        )
        
        return BuildPumpProbeMasksOutput(
            signal_mask=final_signal_mask,
            background_mask=background_mask,
            intermediate_masks=intermediate_masks
        )
    
    def _validate_inputs(
        self,
        input_data: BuildPumpProbeMasksInput
    ) -> None:
        """Validate input data dimensions."""
        p_values = input_data.p_values_output.p_values
        histograms = input_data.histogram_output.histograms
        
        if p_values.shape != histograms.shape[1:]:
            raise ValueError(
                f"P-value shape {p_values.shape} doesn't match "
                f"histogram shape {histograms.shape[1:]}"
            )
            
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
            
        if signal_size > total_size * 0.5:
            warnings.warn(
                f"Signal mask covers {signal_size/total_size:.1%} of image",
                RuntimeWarning
            )
        if background_size > total_size * 0.5:
            warnings.warn(
                f"Background mask covers {background_size/total_size:.1%} of image",
                RuntimeWarning
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
        """Generate diagnostic plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Plot mask generation stages
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Signal Mask Generation Stages')
        
        stages = [
            ('Initial Threshold', output.intermediate_masks.initial),
            ('Signal Clusters', output.intermediate_masks.roi_masked),
            ('Filtered', output.intermediate_masks.filtered),
            ('Final Signal', output.intermediate_masks.final)
        ]
        
        for ax, (title, mask) in zip(axes.flat, stages):
            ax.imshow(mask, cmap='RdBu')
            ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'mask_generation_stages.png')
        plt.close()
        
        # 2. Plot final masks
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(output.signal_mask, cmap='RdBu')
        ax1.set_title('Signal Mask')
        
        ax2.imshow(output.background_mask, cmap='RdBu')
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
        ax.contour(output.background_mask, colors='r', levels=[0.5],
                  linewidths=2, label='Background Mask')
        
        ax.set_title('Distance from Signal to Background')
        plt.tight_layout()
        plt.savefig(save_dir / 'mask_distance.png')
        plt.close()
