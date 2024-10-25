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

from pathlib import Path
from typing import Dict, List, Tuple, Any, NamedTuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import logging

from btx.processing.btx_types import (
    PumpProbeAnalysisInput,
    PumpProbeAnalysisOutput,
)

# Configure logging
logger = logging.getLogger('PumpProbeAnalysis')

class DelayData(NamedTuple):
    """Container for frames and I0 values at a specific delay."""
    frames: np.ndarray
    I0: np.ndarray

class PumpProbeAnalysis:
    """Analyze pump-probe time series data with proper error propagation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pump-probe analysis task.
        
        Args:
            config: Dictionary containing:
                - pump_probe_analysis.min_count: Minimum frames per delay bin
                - pump_probe_analysis.significance_level: P-value threshold
                - pump_probe_analysis.Emin: Minimum energy threshold (keV)
                - pump_probe_analysis.Emax: Maximum energy threshold (keV)
        """
        self.config = config
        
        # Set defaults
        if 'pump_probe_analysis' not in self.config:
            self.config['pump_probe_analysis'] = {}
            
        analysis_config = self.config['pump_probe_analysis']
        if 'min_count' not in analysis_config:
            analysis_config['min_count'] = 2
        if 'significance_level' not in analysis_config:
            analysis_config['significance_level'] = 0.05
        if 'Emin' not in analysis_config:
            analysis_config['Emin'] = 7.0
        if 'Emax' not in analysis_config:
            analysis_config['Emax'] = float('inf')

    def _make_energy_filter(self, frames: np.ndarray) -> np.ndarray:
        """Create energy filter mask based on config parameters.
        
        Args:
            frames: Input frames to filter
            
        Returns:
            Boolean mask for energy filtering
        """
        Emin = self.config['pump_probe_analysis']['Emin']
        Emax = self.config['pump_probe_analysis']['Emax']
        return (frames >= Emin) & (frames <= Emax)

    def _group_by_delay(
        self, 
        input_data: PumpProbeAnalysisInput
    ) -> Tuple[Dict[float, DelayData], Dict[float, DelayData]]:
        """Group frames by delay value with improved binning logic.
        
        Args:
            input_data: Input data containing delays and frames
            
        Returns:
            Tuple of (laser_on_groups, laser_off_groups) dictionaries
        """
        min_count = self.config['pump_probe_analysis']['min_count']
        delays = input_data.load_data_output.binned_delays
        time_bin = float(self.config['load_data']['time_bin'])
        
        # Get unique delays more robustly
        sorted_delays = np.sort(np.unique(delays))
        logger.debug(f"Found {len(sorted_delays)} unique delay values")
        
        # Group delays that are within time_bin/10 of each other
        unique_delays = []
        current_group = [sorted_delays[0]]
        
        for d in sorted_delays[1:]:
            if np.abs(d - current_group[0]) <= time_bin/10:
                current_group.append(d)
            else:
                unique_delays.append(np.mean(current_group))
                current_group = [d]
        
        if current_group:
            unique_delays.append(np.mean(current_group))
        
        unique_delays = np.array(unique_delays)
        logger.debug(f"Grouped into {len(unique_delays)} delay points")
        
        # Group frames by delay
        stacks_on = {}
        stacks_off = {}
        
        for delay in unique_delays:
            delay_mask = np.isclose(delays, delay, rtol=1e-5, atol=time_bin/10)
            
            # Split into on/off
            on_mask = delay_mask & input_data.load_data_output.laser_on_mask
            off_mask = delay_mask & input_data.load_data_output.laser_off_mask
            
            n_on = np.sum(on_mask)
            n_off = np.sum(off_mask)
            
            if n_on >= min_count and n_off >= min_count:
                logger.debug(f"Delay {delay:.3f}ps: {n_on} on, {n_off} off frames")
                stacks_on[delay] = DelayData(
                    frames=input_data.load_data_output.data[on_mask],
                    I0=input_data.load_data_output.I0[on_mask]
                )
                stacks_off[delay] = DelayData(
                    frames=input_data.load_data_output.data[off_mask],
                    I0=input_data.load_data_output.I0[off_mask]
                )
            else:
                logger.debug(f"Skipping delay {delay:.3f}ps: insufficient frames")
        
        if not stacks_on:
            raise ValueError(
                f"No delay groups met the minimum frame count requirement of {min_count}"
            )
        
        return stacks_on, stacks_off

    def _calculate_signals(
        self,
        frames: np.ndarray,
        signal_mask: np.ndarray,
        bg_mask: np.ndarray,
        I0: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate normalized signals and their uncertainties with proper error propagation.
        
        Args:
            frames: Raw frame data
            signal_mask: Signal region mask
            bg_mask: Background region mask
            I0: I0 values for each frame
            
        Returns:
            Tuple of (normalized_signal, normalized_background, variance)
        """
        # Apply energy filter
        energy_mask = self._make_energy_filter(frames)
        filtered_frames = frames * energy_mask
        
        # Calculate per-frame sums
        signal_sums = np.sum(filtered_frames * signal_mask[None, :, :], axis=(1,2))
        bg_sums = np.sum(filtered_frames * bg_mask[None, :, :], axis=(1,2))
        
        # Scale background by mask sizes
        scale_factor = np.sum(signal_mask) / np.sum(bg_mask)
        bg_sums *= scale_factor
        
        # Normalize each frame by its I0
        norm_signal = signal_sums / I0
        norm_bg = bg_sums / I0
        
        # Calculate difference for each frame
        norm_diff = norm_signal - norm_bg
        
        # Calculate statistics
        n_frames = len(frames)
        signal_mean = np.mean(norm_signal)
        bg_mean = np.mean(norm_bg)
        
        # Calculate variance of the normalized difference
        # Using ddof=1 for unbiased variance estimate
        variance = np.var(norm_diff, ddof=1) / n_frames  # Standard error
        
        return signal_mean, bg_mean, variance

    def process(
        self,
        config: Dict[str, Any],
        load_data_output: LoadDataOutput,
        masks_output: BuildPumpProbeMasksOutput
    ) -> PumpProbeAnalysisOutput:
        """Process pump-probe analysis with raw inputs.
        
        Args:
            config: Configuration dictionary
            load_data_output: Output from LoadData task
            masks_output: Output from BuildPumpProbeMasks task
            
        Returns:
            PumpProbeAnalysisOutput with calculated signals and uncertainties
        """
        input_data = PumpProbeAnalysisInput(
            config=config,
            load_data_output=load_data_output,
            masks_output=masks_output
        )
        return self.run(input_data)

    def run(self, input_data: PumpProbeAnalysisInput) -> PumpProbeAnalysisOutput:
        """Run pump-probe analysis with corrected error propagation.
        
        Args:
            input_data: Input data containing frames and masks
            
        Returns:
            PumpProbeAnalysisOutput with calculated signals and uncertainties
            
        .. deprecated:: 2.0.0
           Use process() instead. This method will be removed in a future version.
        """
        import warnings
        warnings.warn(
            "The run() method is deprecated. Use process() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Store masks for diagnostics
        self.signal_mask = input_data.masks_output.signal_mask
        self.bg_mask = input_data.masks_output.background_mask
        
        # Group frames by delay
        self.stacks_on, self.stacks_off = self._group_by_delay(input_data)
        
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
            
            # Calculate signals and uncertainties
            signal_on, bg_on, var_on = self._calculate_signals(
                on_data.frames, self.signal_mask, self.bg_mask, on_data.I0
            )
            
            signal_off, bg_off, var_off = self._calculate_signals(
                off_data.frames, self.signal_mask, self.bg_mask, off_data.I0
            )
            
            # Store results (already normalized by I0)
            delays.append(delay)
            signals_on.append(signal_on - bg_on)
            signals_off.append(signal_off - bg_off)
            std_devs_on.append(np.sqrt(var_on))
            std_devs_off.append(np.sqrt(var_off))
        
        # Calculate mean I0 values
        mean_I0_on = np.mean([
            np.mean(self.stacks_on[delay].I0) for delay in delays
        ])
        mean_I0_off = np.mean([
            np.mean(self.stacks_off[delay].I0) for delay in delays
        ])
        
        # Convert to arrays
        delays = np.array(delays)
        signals_on = np.array(signals_on)
        signals_off = np.array(signals_off)
        std_devs_on = np.array(std_devs_on)
        std_devs_off = np.array(std_devs_off)
        
        # Calculate p-values comparing on/off signals
        z_scores = (signals_on - signals_off) / np.sqrt(std_devs_on**2 + std_devs_off**2)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
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

    def plot_diagnostics(
        self,
        output: PumpProbeAnalysisOutput,
        save_dir: Path
    ) -> None:
        """Generate diagnostic plots with proper infinity handling."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # print("\n=== Signal Shape Debug ===")
        # print(f"signals_on shape: {output.signals_on.shape}")
        # print(f"signals_off shape: {output.signals_off.shape}")
        # print(f"delays: {output.delays}")
        # print("\nFrame counts per delay:")
        # for delay, (n_on, n_off) in output.n_frames_per_delay.items():
        #     print(f"Delay {delay:.3f}: {n_on} on, {n_off} off")
        
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
            
            # Leave second subplot empty
            axes[2,1].set_visible(False)
            
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
