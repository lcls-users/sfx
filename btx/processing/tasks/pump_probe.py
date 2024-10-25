from pathlib import Path
from typing import Dict, List, Tuple, Any, NamedTuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from btx.processing.btx_types import (
    PumpProbeAnalysisInput,
    PumpProbeAnalysisOutput,
)

import warnings
import logging

# Configure logging to file instead of console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pump_probe_analysis.log'),
        logging.NullHandler()  # Prevents logging to console
    ]
)
logger = logging.getLogger('PumpProbeAnalysis')

# Suppress specific numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*P-value underflow.*')

class DelayData(NamedTuple):
    """Container for frames and I0 values at a specific delay."""
    frames: np.ndarray
    I0: np.ndarray

class PumpProbeAnalysis:
    """Analyze pump-probe time series data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pump-probe analysis task.
        
        Args:
            config: Dictionary containing:
                - pump_probe_analysis.min_count: Minimum frames per delay bin
                - pump_probe_analysis.significance_level: P-value threshold
        """
        self.config = config
        
        # Set defaults
        if 'pump_probe_analysis' not in self.config:
            self.config['pump_probe_analysis'] = {}
            
        analysis_config = self.config['pump_probe_analysis']
        if 'min_count' not in analysis_config:
            analysis_config['min_count'] = 1
        if 'significance_level' not in analysis_config:
            analysis_config['significance_level'] = 0.05
        if 'Emin' not in analysis_config:
            analysis_config['Emin'] = 7.0  # keV
        if 'Emax' not in analysis_config:
            analysis_config['Emax'] = float('inf')  # keV

    def _group_by_delay(
        self, 
        input_data: PumpProbeAnalysisInput
    ) -> Tuple[Dict[float, DelayData], Dict[float, DelayData]]:
        """Group frames by delay value.
        
        Returns:
            Tuple of (laser_on_groups, laser_off_groups) dictionaries mapping 
            delays to frame data
        """
        min_count = self.config['pump_probe_analysis']['min_count']
        delays = input_data.load_data_output.binned_delays
        
        # Use the unique binned delays directly - they were already properly binned in LoadData
        unique_delays = np.unique(delays)
        
        
        # Group frames by bin center
        stacks_on = {}
        stacks_off = {}
        
        for delay in unique_delays:
            # Find frames matching this delay exactly
            delay_mask = delays == delay
            
            # Split into on/off
            on_mask = delay_mask & input_data.load_data_output.laser_on_mask
            off_mask = delay_mask & input_data.load_data_output.laser_off_mask
            
            n_on = np.sum(on_mask)
            n_off = np.sum(off_mask)
            
            
            # Only include groups with sufficient frames
            if n_on >= min_count and n_off >= min_count:
                stacks_on[delay] = DelayData(
                    frames=input_data.load_data_output.data[on_mask],
                    I0=input_data.load_data_output.I0[on_mask]
                )
                stacks_off[delay] = DelayData(
                    frames=input_data.load_data_output.data[off_mask],
                    I0=input_data.load_data_output.I0[off_mask]
                )
        
        
        if not stacks_on:
            plt.figure(figsize=(10, 6))
            plt.hist(delays, bins=100)
            plt.xlabel('Delay (ps)')
            plt.ylabel('Count')
            plt.title('Delay Distribution - Empty Groups')
            save_dir = Path("processing/temp/diagnostic_plots")
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / 'delay_distribution_empty.png')
            plt.close()
            raise ValueError(
                f"No delay groups met the minimum frame count requirement of {min_count}.\n"
                f"Check delay_distribution_empty.png for diagnostics."
            )
        
        return stacks_on, stacks_off

    def _make_energy_filter(self, frames: np.ndarray) -> np.ndarray:
        """Create energy filter mask based on config parameters."""
        Emin = self.config['pump_probe_analysis']['Emin']
        Emax = self.config['pump_probe_analysis']['Emax']
        
        # Create energy window
        energy_mask = (frames >= Emin) & (frames <= Emax)
        return energy_mask

    def _calculate_signals(
        self,
        frames: np.ndarray,
        signal_mask: np.ndarray,
        bg_mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate signal, background and total variance for a group of frames."""
        # Apply energy filter to each frame
        energy_mask = self._make_energy_filter(frames)
        filtered_frames = frames * energy_mask
        
        # Calculate per-frame sums
        signal_sums = np.sum(filtered_frames * signal_mask[None, :, :], axis=(1,2))
        bg_sums = np.sum(filtered_frames * bg_mask[None, :, :], axis=(1,2))
        
        # Scale background by mask sizes
        scale_factor = np.sum(signal_mask) / np.sum(bg_mask)
        bg_sums *= scale_factor
        
        # Calculate means
        signal_mean = np.mean(signal_sums)
        bg_mean = np.mean(bg_sums)
        
        # Calculate variances with detailed prints
        n_frames = len(frames)
        signal_var = np.var(signal_sums, ddof=1) / n_frames
        bg_var = np.var(bg_sums, ddof=1) / n_frames
        
        return signal_mean, bg_mean, signal_var + bg_var

    def run(self, input_data: PumpProbeAnalysisInput) -> PumpProbeAnalysisOutput:
        """Run pump-probe analysis with additional error diagnostics."""
        # Store masks and data for diagnostics
        self.signal_mask = input_data.masks_output.signal_mask
        self.bg_mask = input_data.masks_output.background_mask
        
        # Group frames by delay
        self.stacks_on, self.stacks_off = self._group_by_delay(input_data)
        
        if not self.stacks_on:
            raise ValueError("No delay groups met the minimum frame count requirement")
        
        # Process each delay group
        delays = []
        signals_on = []
        signals_off = []
        std_devs_on = []
        std_devs_off = []
        n_frames = {}
        
        for delay in sorted(self.stacks_on.keys()):
            # Get frame stacks
            on_data = self.stacks_on[delay]
            off_data = self.stacks_off[delay]
            
            # Store frame counts
            n_frames[delay] = (len(on_data.frames), len(off_data.frames))
            
            signal_on, bg_on, var_on = self._calculate_signals(
                on_data.frames, self.signal_mask, self.bg_mask
            )
            
            signal_off, bg_off, var_off = self._calculate_signals(
                off_data.frames, self.signal_mask, self.bg_mask
            )
            
            # Normalize by I0 after background subtraction
            I0_mean_on = np.mean(on_data.I0)
            I0_mean_off = np.mean(off_data.I0)
            
            norm_signal_on = (signal_on - bg_on) / I0_mean_on
            norm_signal_off = (signal_off - bg_off) / I0_mean_off
            
            # Propagate errors through I0 normalization
            std_dev_on = np.sqrt(var_on) / I0_mean_on
            std_dev_off = np.sqrt(var_off) / I0_mean_off
            
            delays.append(delay)
            signals_on.append(norm_signal_on)
            signals_off.append(norm_signal_off)
            std_devs_on.append(std_dev_on)
            std_devs_off.append(std_dev_off)
        
        # Convert to arrays
        delays = np.array(delays)
        signals_on = np.array(signals_on)
        signals_off = np.array(signals_off)
        std_devs_on = np.array(std_devs_on)
        std_devs_off = np.array(std_devs_off)
        
        print("\nFinal error statistics:")
        print(f"Mean signal ON std dev: {np.mean(std_devs_on):.3f}")
        print(f"Mean signal OFF std dev: {np.mean(std_devs_off):.3f}")
        print(f"Relative errors ON: {np.mean(std_devs_on/np.abs(signals_on))*100:.1f}%")
        print(f"Relative errors OFF: {np.mean(std_devs_off/np.abs(signals_off))*100:.1f}%")
        
        # Calculate p-values
        p_values = self._calculate_p_values(
            signals_on, signals_off,
            std_devs_on, std_devs_off
        )
        
        # Calculate mean I0 values across all delays
        mean_I0_on = np.mean([
            np.mean(self.stacks_on[delay].I0) for delay in delays
        ])
        mean_I0_off = np.mean([
            np.mean(self.stacks_off[delay].I0) for delay in delays
        ])
        
        return PumpProbeAnalysisOutput(
            delays=delays,
            signals_on=signals_on,
            signals_off=signals_off,
            std_devs_on=std_devs_on,
            std_devs_off=std_devs_off,
            p_values=p_values,
            log_p_values=-np.log10(p_values),
            mean_I0_on=mean_I0_on,
            mean_I0_off=mean_I0_off,
            n_frames_per_delay=n_frames
        )
    def _calculate_p_values(
        self,
        signals_on: np.ndarray,
        signals_off: np.ndarray,
        std_devs_on: np.ndarray,
        std_devs_off: np.ndarray
    ) -> np.ndarray:
        """Calculate p-values comparing on/off signals using two-tailed t-test."""
        delta_signals = signals_on - signals_off  
        combined_stds = np.sqrt(std_devs_on**2 + std_devs_off**2)
        z_scores = delta_signals / combined_stds
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        
        # Ensure p_values are numeric, replace any invalid values with 1.0
        p_values = np.array([p if isinstance(p, (int, float)) else 1.0 for p in p_values])
        return p_values

    def plot_diagnostics(
        self,
        output: PumpProbeAnalysisOutput,
        save_dir: Path
    ) -> None:
        """Generate diagnostic plots with proper infinity handling."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create four-panel overview figure
        fig = plt.figure(figsize=(16, 16))
        
        # Get all frames from all delays for intensity maps
        all_frames = np.concatenate([
            np.concatenate([self.stacks_on[d].frames, self.stacks_off[d].frames])
            for d in output.delays
        ])
        
        # 1. Total counts map (top left)
        ax1 = fig.add_subplot(221)
        total_counts = np.sum(all_frames, axis=0)
        im1 = ax1.imshow(total_counts, origin='lower', cmap='viridis')
        ax1.set_title('Total Counts Map')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Energy filtered map (top right)
        ax2 = fig.add_subplot(222)
        energy_mask = self._make_energy_filter(all_frames)
        filtered_counts = np.sum(all_frames * energy_mask, axis=0)
        im2 = ax2.imshow(filtered_counts, origin='lower', cmap='viridis')
        ax2.set_title(f'Energy Filtered Map (Emin={self.config["pump_probe_analysis"]["Emin"]}, Emax={self.config["pump_probe_analysis"]["Emax"]})')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Time traces with error bars (bottom left)
        ax3 = fig.add_subplot(223)
        ax3.errorbar(output.delays, output.signals_on,
                    yerr=output.std_devs_on, fmt='rs-',
                    label='Laser On', capsize=3)
        ax3.errorbar(output.delays, output.signals_off,
                    yerr=output.std_devs_off, fmt='ks-',
                    label='Laser Off', capsize=3, alpha=0.5)
        ax3.set_xlabel('Time Delay (ps)')
        ax3.set_ylabel('Normalized Signal')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Statistical significance (bottom right)
        ax4 = fig.add_subplot(224)
        
        # Convert p-values to log scale with capped infinities
        max_log_p = 16  # Maximum value to show on plot
        log_p_values = np.zeros_like(output.p_values)
        
        for i, (delay, p_val) in enumerate(zip(output.delays, output.p_values)):
            if p_val > 0:
                log_p = -np.log10(p_val)
                log_p_values[i] = log_p
            else:
                log_p_values[i] = max_log_p
        
        # Create scatter plot with processed values
        scatter = ax4.scatter(output.delays, log_p_values,
                             color='red', label='-log(p-value)')
        
        # Add significance line
        sig_level = self.config['pump_probe_analysis']['significance_level']
        sig_line = -np.log10(sig_level)
        ax4.axhline(y=sig_line, color='k', linestyle='--',
                    label=f'p={sig_level}')
        
        # Set y-axis limits explicitly
        ax4.set_ylim(0, max_log_p * 1.1)  # Add 10% padding above max
        
        ax4.set_xlabel('Time Delay (ps)')
        ax4.set_ylabel('-log10(P-value)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'overview_diagnostics.png')
        plt.close()

        # Plot detailed diagnostics for selected delay points
        delay_indices = [0, len(output.delays)//2, -1]  # First, middle, and last delays
        for idx in delay_indices:
            delay = output.delays[idx]
            
            # Get data from stored stacks
            frames_on = self.stacks_on[delay].frames
            frames_off = self.stacks_off[delay].frames
            I0_on = self.stacks_on[delay].I0
            I0_off = self.stacks_off[delay].I0
            
            # Calculate statistics
            signal_sums_on = np.sum(frames_on * self.signal_mask[None, :, :], axis=(1,2))
            signal_sums_off = np.sum(frames_off * self.signal_mask[None, :, :], axis=(1,2))
            bg_sums_on = np.sum(frames_on * self.bg_mask[None, :, :], axis=(1,2))
            bg_sums_off = np.sum(frames_off * self.bg_mask[None, :, :], axis=(1,2))
            
            scale_factor = np.sum(self.signal_mask) / np.sum(self.bg_mask)
            net_signal_on = (signal_sums_on - scale_factor * bg_sums_on) / np.mean(I0_on)
            net_signal_off = (signal_sums_off - scale_factor * bg_sums_off) / np.mean(I0_off)
            
            # Create detailed diagnostic figure
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            fig.suptitle(f'Detailed Diagnostics for Delay {delay:.2f} ps', fontsize=16)
            
            # Raw signal distributions
            axes[0,0].hist(signal_sums_on, bins='auto', alpha=0.5, label='Laser On')
            axes[0,0].hist(signal_sums_off, bins='auto', alpha=0.5, label='Laser Off')
            axes[0,0].set_title('Raw Signal Distributions')
            axes[0,0].set_xlabel('Integrated Signal')
            axes[0,0].set_ylabel('Count')
            axes[0,0].legend()
            
            # Background distributions
            axes[0,1].hist(bg_sums_on * scale_factor, bins='auto', alpha=0.5, label='Laser On')
            axes[0,1].hist(bg_sums_off * scale_factor, bins='auto', alpha=0.5, label='Laser Off')
            axes[0,1].set_title('Scaled Background Distributions')
            axes[0,1].set_xlabel('Integrated Background (scaled)')
            axes[0,1].set_ylabel('Count')
            axes[0,1].legend()
            
            # Frame-to-frame variations
            axes[1,0].plot(np.arange(len(signal_sums_on)), signal_sums_on, 'r.', label='Signal On')
            axes[1,0].plot(np.arange(len(signal_sums_off)), signal_sums_off, 'b.', label='Signal Off')
            axes[1,0].set_title('Frame-to-Frame Signal Variation')
            axes[1,0].set_xlabel('Frame Number')
            axes[1,0].set_ylabel('Integrated Signal')
            axes[1,0].legend()
            
            # I0 variations
            axes[1,1].plot(np.arange(len(I0_on)), I0_on, 'r.', label='I0 On')
            axes[1,1].plot(np.arange(len(I0_off)), I0_off, 'b.', label='I0 Off')
            axes[1,1].set_title('Frame-to-Frame I0 Variation')
            axes[1,1].set_xlabel('Frame Number')
            axes[1,1].set_ylabel('I0')
            axes[1,1].legend()
            
            # Net signal distributions
            axes[2,0].hist(net_signal_on, bins='auto', alpha=0.5, label='Laser On')
            axes[2,0].hist(net_signal_off, bins='auto', alpha=0.5, label='Laser Off')
            axes[2,0].set_title('Normalized Net Signal Distributions')
            axes[2,0].set_xlabel('Net Signal (normalized)')
            axes[2,0].set_ylabel('Count')
            axes[2,0].legend()
            
            # QQ plot of differences
            stats.probplot(net_signal_on - net_signal_off, dist="norm", plot=axes[2,1])
            axes[2,1].set_title('Q-Q Plot of Signal Differences')
            
            # Add statistical information
            stats_text = (
                f'Statistics:\n'
                f'N frames (on/off): {len(signal_sums_on)}/{len(signal_sums_off)}\n'
                f'Signal mean ± SE (on): {np.mean(net_signal_on):.2e} ± {np.std(net_signal_on)/np.sqrt(len(net_signal_on)):.2e}\n'
                f'Signal mean ± SE (off): {np.mean(net_signal_off):.2e} ± {np.std(net_signal_off)/np.sqrt(len(net_signal_off)):.2e}\n'
                f'Signal CV (on/off): {np.std(net_signal_on)/np.mean(net_signal_on):.2%}/{np.std(net_signal_off)/np.mean(net_signal_off):.2%}\n'
                f'P-value: {output.p_values[idx]:.2e}\n'
                f'Log p-value: {log_p_values[idx]:.1f}\n'
                f'Z-score: {(np.mean(net_signal_on) - np.mean(net_signal_off))/np.sqrt(output.std_devs_on[idx]**2 + output.std_devs_off[idx]**2):.2f}'
            )
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, family='monospace')
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(save_dir / f'detailed_diagnostics_delay_{delay:.2f}ps.png')
            plt.close()
