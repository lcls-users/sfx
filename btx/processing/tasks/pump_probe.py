from pathlib import Path
from typing import Dict, List, Tuple, Any, NamedTuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from btx.processing.btx_types import (
    PumpProbeAnalysisInput,
    PumpProbeAnalysisOutput,
)

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
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'pump_probe_analysis' not in self.config:
            raise ValueError("Missing 'pump_probe_analysis' section in config")
            
        required_params = ['min_count', 'significance_level']
        for param in required_params:
            if param not in self.config['pump_probe_analysis']:
                raise ValueError(f"Missing required parameter: {param}")
        
        min_count = self.config['pump_probe_analysis']['min_count']
        if not isinstance(min_count, (int, np.integer)) or min_count < 1:
            raise ValueError("min_count must be a positive integer")
            
        sig_level = self.config['pump_probe_analysis']['significance_level']
        if not isinstance(sig_level, (int, float)) or not 0 < sig_level < 1:
            raise ValueError("significance_level must be between 0 and 1")

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
        
        print(f"\nDelay grouping diagnostics:")
        print(f"Delay range: {delays.min():.1f} to {delays.max():.1f} ps")
        print(f"Number of bins: {len(unique_delays)}")
        
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
            
            print(f"Delay {delay:.1f} ps: {n_on} on frames, {n_off} off frames")
            
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
        
        print(f"Created {len(stacks_on)} delay groups meeting minimum count requirement")
        
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

    def _calculate_signals(
        self,
        frames: np.ndarray,
        signal_mask: np.ndarray,
        bg_mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate signal, background and total variance for a group of frames.
        
        For each frame:
        1. Sum over signal region
        2. Sum over background region (scaled)
        3. Calculate mean and standard error across frames
        
        Args:
            frames: Array of frames
            signal_mask: Signal region mask
            bg_mask: Background region mask
            
        Returns:
            Tuple of (signal, background, total_variance)
        """
        # Calculate per-frame sums
        signal_sums = np.sum(frames * signal_mask[None, :, :], axis=(1,2))
        bg_sums = np.sum(frames * bg_mask[None, :, :], axis=(1,2))
        
        # Scale background by mask sizes
        scale_factor = np.sum(signal_mask) / np.sum(bg_mask)
        bg_sums *= scale_factor
        
        # Calculate means and standard errors
        signal_mean = np.mean(signal_sums)
        bg_mean = np.mean(bg_sums)
        
        # Use standard error of the mean for each component
        n_frames = len(frames)
        signal_var = np.var(signal_sums, ddof=1) / n_frames  # ddof=1 for sample variance
        bg_var = np.var(bg_sums, ddof=1) / n_frames
        
        return signal_mean, bg_mean, signal_var + bg_var

    def run(self, input_data: PumpProbeAnalysisInput) -> PumpProbeAnalysisOutput:
        """Run pump-probe analysis."""
        # Group frames by delay
        stacks_on, stacks_off = self._group_by_delay(input_data)
        
        if not stacks_on:
            raise ValueError("No delay groups met the minimum frame count requirement")
        
        # Process each delay group
        delays = []
        signals_on = []
        signals_off = []
        std_devs_on = []
        std_devs_off = []
        n_frames = {}
        
        signal_mask = input_data.masks_output.signal_mask
        bg_mask = input_data.masks_output.background_mask
        
        for delay in sorted(stacks_on.keys()):
            # Get frame stacks
            on_data = stacks_on[delay]
            off_data = stacks_off[delay]
            
            # Store frame counts
            n_frames[delay] = (len(on_data.frames), len(off_data.frames))
            
            # Calculate signals and background
            signal_on, bg_on, var_on = self._calculate_signals(
                on_data.frames, signal_mask, bg_mask
            )
            signal_off, bg_off, var_off = self._calculate_signals(
                off_data.frames, signal_mask, bg_mask
            )
            
            # Normalize by I0 after background subtraction
            norm_signal_on = (signal_on - bg_on) / np.mean(on_data.I0)
            norm_signal_off = (signal_off - bg_off) / np.mean(off_data.I0)
            
            # Propagate errors through I0 normalization
            std_dev_on = np.sqrt(var_on) / np.mean(on_data.I0)
            std_dev_off = np.sqrt(var_off) / np.mean(off_data.I0)
            
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
        
        # Calculate p-values
        p_values = self._calculate_p_values(
            signals_on, signals_off,
            std_devs_on, std_devs_off
        )
        
        # Calculate mean I0 values across all delays
        mean_I0_on = np.mean([
            np.mean(stacks_on[delay].I0) for delay in delays
        ])
        mean_I0_off = np.mean([
            np.mean(stacks_off[delay].I0) for delay in delays
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
        delta_signals = signals_on - signals_off  # Note: no abs() here for proper two-sided test
        combined_stds = np.sqrt(std_devs_on**2 + std_devs_off**2)
        z_scores = delta_signals / combined_stds
        # Two-tailed test - use absolute z-score
        return 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    def plot_diagnostics(
        self,
        output: PumpProbeAnalysisOutput,
        save_dir: Path
    ) -> None:
        """Generate diagnostic plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Time traces with error bars
        ax1 = fig.add_subplot(221)
        ax1.errorbar(output.delays, output.signals_on,
                    yerr=output.std_devs_on, fmt='rs-',
                    label='Laser On', capsize=3)
        ax1.errorbar(output.delays, output.signals_off,
                    yerr=output.std_devs_off, fmt='ks-',
                    label='Laser Off', capsize=3, alpha=0.5)
        ax1.set_xlabel('Time Delay (ps)')
        ax1.set_ylabel('Normalized Signal')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Difference signal
        ax2 = fig.add_subplot(222)
        diff_signal = output.signals_on - output.signals_off
        diff_std = np.sqrt(output.std_devs_on**2 + output.std_devs_off**2)
        ax2.errorbar(output.delays, diff_signal,
                    yerr=diff_std, fmt='bo-',
                    label='On - Off', capsize=3)
        ax2.set_xlabel('Time Delay (ps)')
        ax2.set_ylabel('Signal Difference')
        ax2.grid(True)
        
        # 3. P-values
        ax3 = fig.add_subplot(223)
        ax3.scatter(output.delays, output.log_p_values,
                   color='red', label='-log(p-value)')
        sig_level = self.config['pump_probe_analysis']['significance_level']
        sig_line = -np.log10(sig_level)
        ax3.axhline(y=sig_line, color='k', linestyle='--',
                   label=f'p={sig_level}')
        ax3.set_xlabel('Time Delay (ps)')
        ax3.set_ylabel('-log(P-value)')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Number of frames per delay
        ax4 = fig.add_subplot(224)
        delays = list(output.n_frames_per_delay.keys())
        n_on = [v[0] for v in output.n_frames_per_delay.values()]
        n_off = [v[1] for v in output.n_frames_per_delay.values()]
        ax4.bar(delays, n_on, alpha=0.5, label='Laser On')
        ax4.bar(delays, n_off, alpha=0.5, label='Laser Off')
        ax4.axhline(y=self.config['pump_probe_analysis']['min_count'],
                   color='r', linestyle='--', label='Min Count')
        ax4.set_xlabel('Time Delay (ps)')
        ax4.set_ylabel('Number of Frames')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'pump_probe_diagnostics.png')
        plt.close()
