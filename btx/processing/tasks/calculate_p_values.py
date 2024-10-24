from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import warnings

from btx.processing.btx_types import CalculatePValuesInput, CalculatePValuesOutput

class CalculatePValues:
    """Calculate p-values from EMD values and null distribution."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize p-value calculation task.
        
        Args:
            config: Dictionary containing:
                - calculate_pvalues.significance_threshold: P-value threshold (default: 0.05)
        """
        self.config = config
        
        # Set defaults
        if 'calculate_pvalues' not in self.config:
            self.config['calculate_pvalues'] = {}
        if 'significance_threshold' not in self.config['calculate_pvalues']:
            self.config['calculate_pvalues']['significance_threshold'] = 0.05

    def _calculate_p_values(
        self,
        emd_values: np.ndarray,
        null_distribution: np.ndarray
    ) -> np.ndarray:
        """Calculate p-values for each pixel.
        
        Args:
            emd_values: 2D array of EMD values
            null_distribution: 1D array of null distribution values
            
        Returns:
            2D array of p-values
        """
        p_values = np.zeros_like(emd_values)
        min_p_value = 1.0 / (len(null_distribution) + 1)
        
        for i in range(emd_values.shape[0]):
            for j in range(emd_values.shape[1]):
                p_value = np.mean(null_distribution >= emd_values[i, j])
                if p_value == 0:
                    warnings.warn(
                        f"P-value underflow at pixel ({i},{j}). "
                        f"Setting to minimum possible value {min_p_value:.2e}",
                        RuntimeWarning
                    )
                    p_value = min_p_value
                p_values[i, j] = p_value
                
        return p_values

    def run(self, input_data: CalculatePValuesInput) -> CalculatePValuesOutput:
        """Run p-value calculation.
        
        Args:
            input_data: CalculatePValuesInput containing EMD values and null distribution
            
        Returns:
            CalculatePValuesOutput containing p-values and derived data
            
        Raises:
            ValueError: If input data is invalid
        """
        emd_values = input_data.emd_output.emd_values
        null_distribution = input_data.emd_output.null_distribution
        
        # Calculate p-values
        p_values = self._calculate_p_values(emd_values, null_distribution)
        
        # Calculate -log10(p-values) for visualization
        # Handle zeros by using minimum possible p-value
        min_p_value = 1.0 / (len(null_distribution) + 1)
        log_p_values = -np.log10(np.maximum(p_values, min_p_value))
        
        # Get significance threshold
        threshold = self.config['calculate_pvalues']['significance_threshold']
        
        # Print some statistics
        n_significant = np.sum(p_values < threshold)
        print(f"Found {n_significant} significant pixels "
              f"(p < {threshold:.3f}, {n_significant/p_values.size:.1%} of total)")
        
        return CalculatePValuesOutput(
            p_values=p_values,
            log_p_values=log_p_values,
            significance_threshold=threshold
        )
        
    def plot_diagnostics(self, output: CalculatePValuesOutput, save_dir: Path) -> None:
        """Generate diagnostic plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. P-value spatial distribution (log scale)
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(output.log_p_values, cmap='viridis')
        ax1.set_title('-log10(P-values)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Binary significance mask
        ax2 = fig.add_subplot(222)
        signif_mask = output.p_values < output.significance_threshold
        im2 = ax2.imshow(signif_mask, cmap='RdBu')
        ax2.set_title(f'Significant Pixels (p < {output.significance_threshold:.3f})')
        plt.colorbar(im2, ax=ax2)
        
        # 3. P-value histogram
        ax3 = fig.add_subplot(223)
        ax3.hist(output.p_values.ravel(), bins=50, density=True)
        ax3.axvline(
            output.significance_threshold,
            color='r',
            linestyle='--',
            label=f'p = {output.significance_threshold:.3f}'
        )
        # Add uniform distribution reference line
        ax3.axhline(1.0, color='k', linestyle=':', label='Uniform')
        ax3.set_xlabel('P-value')
        ax3.set_ylabel('Density')
        ax3.set_title('P-value Distribution')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Q-Q plot
        ax4 = fig.add_subplot(224)
        observed_p = np.sort(output.p_values.ravel())
        expected_p = np.linspace(0, 1, len(observed_p))
        ax4.plot(expected_p, observed_p, 'b.', alpha=0.1)
        ax4.plot([0, 1], [0, 1], 'r--', label='y=x')
        ax4.set_xlabel('Expected P-value')
        ax4.set_ylabel('Observed P-value')
        ax4.set_title('P-value Q-Q Plot')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'calculate_pvalues_diagnostics.png')
        plt.close()
