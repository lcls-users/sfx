from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from btx.processing.types import (
    PumpProbeAnalysisInput,
    PumpProbeAnalysisOutput,
    DelayGroup
)

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

    def _group_by_delay(self, input_data: PumpProbeAnalysisInput) -> List[DelayGroup]:
        """Group frames by delay value using binning."""
        min_count = self.config['pump_probe_analysis']['min_count']
        delays = input_data.load_data_output.binned_delays
        
        # Define bins with reasonable width (e.g., 1 ps)
        bin_width = 1.0  # ps
        min_delay = np.floor(np.min(delays))
        max_delay = np.ceil(np.max(delays))
        bin_edges = np.arange(min_delay - bin_width/2, max_delay + bin_width, bin_width)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        print(f"\nDelay grouping diagnostics:")
        print(f"Delay range: {min_delay:.1f} to {max_delay:.1f} ps")
        print(f"Bin width: {bin_width} ps")
        print(f"Number of bins: {len(bin_centers)}")
        
        # Plot delay distribution
        plt.figure(figsize=(10, 6))
        plt.hist(delays, bins=50, alpha=0.5, label='All delays')
        plt.hist(delays[input_data.load_data_output.laser_on_mask], bins=50, alpha=0.5, label='Laser on')
        plt.hist(delays[input_data.load_data_output.laser_off_mask], bins=50, alpha=0.5, label='Laser off')
        for edge in bin_edges:
            plt.axvline(edge, color='k', linestyle='--', alpha=0.2)
        plt.xlabel('Delay (ps)')
        plt.ylabel('Count')
        plt.title('Delay Distribution')
        plt.legend()
        plt.grid(True)
        
        # Save diagnostic plot
        save_dir = Path("processing/temp/diagnostic_plots")
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'delay_distribution.png')
        plt.close()
        
        delay_groups = []
        for bin_center, left_edge, right_edge in zip(bin_centers, bin_edges[:-1], bin_edges[1:]):
            # Find frames in this bin
            bin_mask = (delays >= left_edge) & (delays < right_edge)
            
            # Split into on/off
            on_mask = bin_mask & input_data.load_data_output.laser_on_mask
            off_mask = bin_mask & input_data.load_data_output.laser_off_mask
            
            n_on = np.sum(on_mask)
            n_off = np.sum(off_mask)
            
            print(f"Delay bin [{left_edge:.1f}, {right_edge:.1f}) ps: {n_on} on frames, {n_off} off frames")
            
            # Only include groups with sufficient frames
            if n_on >= min_count and n_off >= min_count:
                group = DelayGroup(
                    delay=bin_center,
                    on_frames=input_data.load_data_output.data[on_mask],
                    off_frames=input_data.load_data_output.data[off_mask],
                    on_I0=input_data.load_data_output.I0[on_mask],
                    off_I0=input_data.load_data_output.I0[off_mask]
                )
                delay_groups.append(group)
        
        print(f"Created {len(delay_groups)} delay groups meeting minimum count requirement")
        
        if not delay_groups:
            plt.figure(figsize=(10, 6))
            plt.hist(delays, bins=100)
            plt.xlabel('Delay (ps)')
            plt.ylabel('Count')
            plt.title('Delay Distribution - Empty Groups')
            plt.savefig(save_dir / 'delay_distribution_empty.png')
            plt.close()
            raise ValueError(
                f"No delay groups met the minimum frame count requirement of {min_count}.\n"
                f"Check delay_distribution.png and delay_distribution_empty.png for diagnostics."
            )
        
        return sorted(delay_groups, key=lambda x: x.delay)

    def _calculate_signals(
        self,
        frames: np.ndarray,
        I0: np.ndarray,
        signal_mask: np.ndarray,
        bg_mask: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate normalized signal and uncertainty for a group of frames."""
        # Sum over spatial dimensions for each frame
        signal_sum = np.sum(frames * signal_mask[None, :, :], axis=(1, 2))
        bg_sum = np.sum(frames * bg_mask[None, :, :], axis=(1, 2))
        
        # Scale background by mask sizes
        scale_factor = np.sum(signal_mask) / np.sum(bg_mask)
        bg_sum *= scale_factor
        
        # Normalize by I0
        norm_signal = signal_sum / I0
        norm_bg = bg_sum / I0
        
        # Calculate means
        mean_signal = np.mean(norm_signal)
        mean_bg = np.mean(norm_bg)
        
        # Calculate standard errors
        signal_err = np.std(norm_signal) / np.sqrt(len(norm_signal))
        bg_err = np.std(norm_bg) / np.sqrt(len(norm_bg))
        
        # Final signal and error
        final_signal = mean_signal - mean_bg
        final_err = np.sqrt(signal_err**2 + bg_err**2)
        
        return final_signal, final_err

    def run(self, input_data: PumpProbeAnalysisInput) -> PumpProbeAnalysisOutput:
        """Run pump-probe analysis."""
        # Group frames by delay
        delay_groups = self._group_by_delay(input_data)
        
        if not delay_groups:
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
        
        for group in delay_groups:
            delays.append(group.delay)
            n_frames[group.delay] = (len(group.on_frames), len(group.off_frames))
            
            # Calculate signals
            on_signal, on_std = self._calculate_signals(
                group.on_frames, group.on_I0, signal_mask, bg_mask
            )
            off_signal, off_std = self._calculate_signals(
                group.off_frames, group.off_I0, signal_mask, bg_mask
            )
            
            signals_on.append(on_signal)
            signals_off.append(off_signal)
            std_devs_on.append(on_std)
            std_devs_off.append(off_std)
        
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
        
        # Calculate mean I0 values
        mean_I0_on = np.mean([
            np.mean(group.on_I0) for group in delay_groups
        ])
        mean_I0_off = np.mean([
            np.mean(group.off_I0) for group in delay_groups
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
        """Calculate p-values comparing on/off signals."""
        delta_signals = np.abs(signals_on - signals_off)
        combined_stds = np.sqrt(std_devs_on**2 + std_devs_off**2)
        z_scores = delta_signals / combined_stds
        return 2 * (1 - stats.norm.cdf(z_scores))

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
