import pytest
from pathlib import Path
import numpy as np
from scipy import stats, optimize, fft
import matplotlib.pyplot as plt

from btx.processing.tasks.pump_probe import PumpProbeAnalysis
from btx.processing.btx_types import (
    PumpProbeAnalysisInput,
    PumpProbeAnalysisOutput,
    LoadDataOutput,
    BuildPumpProbeMasksOutput
)
from typing import Tuple

def generate_signal_profile(delays: np.ndarray, profile_type: str = 'step', **kwargs) -> np.ndarray:
    """Generate different signal time profiles.
    
    Args:
        delays: Array of delay values
        profile_type: One of ['step', 'exponential', 'gaussian', 'oscillating']
        **kwargs: Profile-specific parameters:
            - step: amplitude, t0
            - exponential: amplitude, t0, decay_time
            - gaussian: amplitude, t0, width
            - oscillating: amplitude, t0, frequency, decay_time
            
    Returns:
        Array of signal values corresponding to delays
    """
    if profile_type == 'step':
        amplitude = kwargs.get('amplitude', 1.0)
        t0 = kwargs.get('t0', 0.0)
        return amplitude * (delays > t0)
        
    elif profile_type == 'exponential':
        amplitude = kwargs.get('amplitude', 1.0)
        t0 = kwargs.get('t0', 0.0)
        decay_time = kwargs.get('decay_time', 5.0)
        signal = np.zeros_like(delays)
        pos_delays = delays > t0
        signal[pos_delays] = amplitude * np.exp(-(delays[pos_delays] - t0)/decay_time)
        return signal
        
    elif profile_type == 'gaussian':
        amplitude = kwargs.get('amplitude', 1.0)
        t0 = kwargs.get('t0', 0.0)
        width = kwargs.get('width', 2.0)
        return amplitude * np.exp(-(delays - t0)**2/(2*width**2))
        
    elif profile_type == 'oscillating':
        amplitude = kwargs.get('amplitude', 1.0)
        t0 = kwargs.get('t0', 0.0)
        frequency = kwargs.get('frequency', 1.0)
        decay_time = kwargs.get('decay_time', 10.0)
        signal = np.zeros_like(delays)
        pos_delays = delays > t0
        signal[pos_delays] = amplitude * np.exp(-(delays[pos_delays] - t0)/decay_time) * \
                            np.sin(2*np.pi*frequency*(delays[pos_delays] - t0))
        return signal
    
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

def generate_synthetic_pump_probe_data(
    n_frames: int = 1000,
    rows: int = 100,
    cols: int = 100,
    delay_range: Tuple[float, float] = (-10, 20),
    n_delay_bins: int = 20,
    base_counts: float = 100.0,  # Base counts for Poisson distribution
    signal_profile: str = 'exponential',
    profile_params: dict = None,
    min_frames_per_bin: int = 20
):
    """Generate synthetic pump-probe data with various signal profiles using Poisson noise.

    Args:
        n_frames: Number of frames to generate
        rows: Number of rows in each frame
        cols: Number of columns in each frame
        delay_range: (min_delay, max_delay) in ps
        n_delay_bins: Number of delay time bins
        base_counts: Base count rate for Poisson distribution
        signal_profile: Type of signal profile to generate
        profile_params: Parameters for the signal profile
        min_frames_per_bin: Minimum frames per delay bin

    Returns:
        Tuple of LoadDataOutput, BuildPumpProbeMasksOutput, and true signal values
    """
    if profile_params is None:
        profile_params = {}
    
    # Create delay bins
    delay_min, delay_max = delay_range
    bin_width = (delay_max - delay_min) / n_delay_bins
    bin_edges = np.linspace(delay_min, delay_max, n_delay_bins + 1)
    delay_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Use specified frames per bin
    frames_per_bin = min_frames_per_bin
    n_frames = frames_per_bin * n_delay_bins
    
    # Create delay assignments using bin centers
    delays = np.repeat(delay_centers, frames_per_bin)
    
    print(f"\nDelay generation diagnostics:")
    print(f"Delay range: {delay_min:.1f} to {delay_max:.1f} ps")
    print(f"Number of bins: {n_delay_bins}")
    print(f"Bin width: {bin_width:.2f} ps")
    print(f"Frames per bin: {frames_per_bin}")

    # Create alternating laser on/off mask
    laser_on = np.zeros(n_frames, dtype=bool)
    for i in range(n_delay_bins):
        start_idx = i * frames_per_bin
        end_idx = (i + 1) * frames_per_bin
        # Make alternating frames laser-on within each delay group
        laser_on[start_idx:end_idx:2] = True
    laser_off = ~laser_on
    
    # Generate frames
    frames = np.zeros((n_frames, rows, cols))
    
    # Create signal and background regions
    signal_mask = np.zeros((rows, cols), dtype=bool)
    signal_mask[20:30, 20:30] = True
    bg_mask = np.zeros((rows, cols), dtype=bool)
    bg_mask[5:15, 5:15] = True
    
    # Generate time-dependent signal
    signal_values = generate_signal_profile(delays, signal_profile, **profile_params)
    
    # Create frames with Poisson noise
    for i in range(n_frames):
        # Base pattern - use Poisson distribution
        lambda_matrix = np.full((rows, cols), base_counts)
        
        # Add signal for laser-on frames
        if laser_on[i]:
            lambda_matrix[20:30, 20:30] *= (1 + signal_values[i])
        
        # Generate Poisson random numbers
        frames[i] = np.random.poisson(lambda_matrix) / 10
    
    # Generate I0 values - also using Poisson for consistency
    I0_base = 1000
    I0 = np.random.poisson(I0_base, n_frames)
    # Add correlation with signal for laser-on frames
    I0[laser_on] = np.random.poisson(I0_base * (1 + 0.1 * signal_values[laser_on]))
    
    # Create outputs
    load_data_output = LoadDataOutput(
        data=frames,
        I0=I0,
        laser_delays=delays,
        binned_delays=delays,  # Using same delays since they're already binned
        laser_on_mask=laser_on,
        laser_off_mask=laser_off
    )
    
    masks_output = BuildPumpProbeMasksOutput(
        signal_mask=signal_mask,
        background_mask=bg_mask,
        intermediate_masks=None
    )
    
    return load_data_output, masks_output, signal_values

def validate_pump_probe_results(
    output: PumpProbeAnalysisOutput,
    signal_profile: str,
    true_params: dict,
    delays: np.ndarray,
    plot_dir: Path
) -> None:
    """Validate pump-probe results against known signal parameters."""
    signal_diff = output.signals_on - output.signals_off
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(output.delays, signal_diff, 
                yerr=np.sqrt(output.std_devs_on**2 + output.std_devs_off**2),
                fmt='o', label='Measured')
    
    # Fit and plot appropriate model
    if signal_profile == 'exponential':
        def exp_model(t, A, tau):
            return A * np.exp(-t/tau)
        
        pos_delays = output.delays > true_params['t0']
        popt, _ = optimize.curve_fit(
            exp_model, 
            output.delays[pos_delays], 
            signal_diff[pos_delays],
            p0=[true_params['amplitude'], true_params['decay_time']]
        )
        
        plt.plot(output.delays[pos_delays], 
                exp_model(output.delays[pos_delays], *popt),
                'r-', label='Fit')
        
        plt.title(f'Exponential Fit\nTrue τ: {true_params["decay_time"]:.1f} ps, '
                 f'Fitted τ: {popt[1]:.1f} ps')
        
    elif signal_profile == 'oscillating':
        def osc_model(t, A, f, tau, phi):
            return A * np.exp(-t/tau) * np.sin(2*np.pi*f*t + phi)
        
        pos_delays = output.delays > true_params['t0']
        popt, _ = optimize.curve_fit(
            osc_model,
            output.delays[pos_delays],
            signal_diff[pos_delays],
            p0=[true_params['amplitude'], 
                true_params['frequency'],
                true_params['decay_time'],
                0]
        )
        
        plt.plot(output.delays[pos_delays],
                osc_model(output.delays[pos_delays], *popt),
                'r-', label='Fit')
                
        plt.title(f'Oscillating Fit\nTrue f: {true_params["frequency"]:.2f} ps⁻¹, '
                 f'Fitted f: {popt[1]:.2f} ps⁻¹')
        
    elif signal_profile == 'gaussian':
        def gauss_model(t, A, t0, w):
            return A * np.exp(-(t - t0)**2/(2*w**2))
            
        popt, _ = optimize.curve_fit(
            gauss_model,
            output.delays,
            signal_diff,
            p0=[true_params['amplitude'],
                true_params['t0'],
                true_params['width']]
        )
        
        plt.plot(output.delays,
                gauss_model(output.delays, *popt),
                'r-', label='Fit')
                
        plt.title(f'Gaussian Fit\nTrue t0: {true_params["t0"]:.1f} ps, '
                 f'Fitted t0: {popt[1]:.1f} ps')
    
    plt.xlabel('Delay (ps)')
    plt.ylabel('Signal Difference')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / f'signal_fit_{signal_profile}.png')
    plt.close()

def test_pump_probe_basic():
    """Basic test of PumpProbeAnalysis configuration."""
    valid_config = {
        'pump_probe_analysis': {
            'min_count': 10,
            'significance_level': 0.05
        }
    }
    
    task = PumpProbeAnalysis(valid_config)
    
    # Test missing configuration
    with pytest.raises(ValueError):
        PumpProbeAnalysis({})
    
    # Test invalid min_count
    invalid_config = {
        'pump_probe_analysis': {
            'min_count': -1,
            'significance_level': 0.05
        }
    }
    with pytest.raises(ValueError):
        PumpProbeAnalysis(invalid_config)

def test_pump_probe_time_dependent():
    """Test PumpProbeAnalysis with various time-dependent signals."""
    test_cases = [
        {
            'profile': 'exponential',
            'params': {
                'amplitude': 0.3,
                't0': 0.0,
                'decay_time': 5.0
            },
            'description': 'Exponential decay'
        },
        {
            'profile': 'oscillating',
            'params': {
                'amplitude': 0.2,
                't0': 0.0,
                'frequency': 0.3,
                'decay_time': 10.0
            },
            'description': 'Damped oscillation'
        },
        {
            'profile': 'gaussian',
            'params': {
                'amplitude': 0.4,
                't0': 5.0,
                'width': 2.0
            },
            'description': 'Gaussian peak'
        }
    ]
    
    config = {
        'pump_probe_analysis': {
            'min_count': 10,
            'significance_level': 0.05
        }
    }
    
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'pump_probe'
    
    for case in test_cases:
        print(f"\nTesting {case['description']}...")
        
        case_dir = save_dir / case['profile']
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        load_data_output, masks_output, true_signal = generate_synthetic_pump_probe_data(
            n_frames=2000,
            n_delay_bins=30,
            noise_level=0.02,
            signal_profile=case['profile'],
            profile_params=case['params'],
            min_frames_per_bin=config['pump_probe_analysis']['min_count'] * 2
        )
        
        # Run analysis
        task = PumpProbeAnalysis(config)
        input_data = PumpProbeAnalysisInput(
            config=config,
            load_data_output=load_data_output,
            masks_output=masks_output
        )
        
        output = task.run(input_data)
        
        # Generate diagnostic plots
        task.plot_diagnostics(output, case_dir)
        
        # Validate results
        validate_pump_probe_results(
            output,
            case['profile'],
            case['params'],
            load_data_output.binned_delays,
            case_dir
        )
        
        # Profile-specific validations
        if case['profile'] == 'exponential':
            # Check decay time within 30%
            pos_delays = output.delays > case['params']['t0']
            signal_diff = output.signals_on[pos_delays] - output.signals_off[pos_delays]
            
            def exp_decay(t, A, tau):
                return A * np.exp(-t/tau)
            
            popt, _ = optimize.curve_fit(exp_decay, 
                                       output.delays[pos_delays], 
                                       signal_diff,
                                       p0=[case['params']['amplitude'],
                                           case['params']['decay_time']])
            fitted_tau = popt[1]
            true_tau = case['params']['decay_time']
            
            assert np.abs(fitted_tau - true_tau) < true_tau * 0.3, \
                f"Fitted decay time {fitted_tau:.1f} differs from true value {true_tau:.1f}"
                
        elif case['profile'] == 'oscillating':
            # Check frequency using FFT
            pos_delays = output.delays > case['params']['t0']
            signal_diff = output.signals_on[pos_delays] - output.signals_off[pos_delays]
            
            # Simple FFT analysis
            yf = fft.fft(signal_diff)
            dt = np.mean(np.diff(output.delays[pos_delays]))
            xf = fft.fftfreq(len(signal_diff), dt)
            
            # Find dominant frequency
            peak_freq = np.abs(xf[np.argmax(np.abs(yf[1:]) + 1)])
            true_freq = case['params']['frequency']
            
            assert np.abs(peak_freq - true_freq) < true_freq * 0.3, \
                f"Fitted frequency {peak_freq:.2f} differs from true value {true_freq:.2f}"
                
        elif case['profile'] == 'gaussian':
            # Check peak position
            signal_diff = output.signals_on - output.signals_off
            peak_idx = np.argmax(signal_diff)
            peak_time = output.delays[peak_idx]
            true_t0 = case['params']['t0']
            
            assert np.abs(peak_time - true_t0) < 2.0, \
                f"Peak time {peak_time:.1f} differs from true value {true_t0:.1f}"
        
        print(f"Generated plots for {case['description']} in: {case_dir}")

def test_pump_probe_statistics():
    """Test statistical properties of the pump-probe analysis."""
    # Generate null data (no signal)
    config = {
        'pump_probe_analysis': {
            'min_count': 10,
            'significance_level': 0.05
        }
    }
    
    load_data_output, masks_output, _ = generate_synthetic_pump_probe_data(
        n_frames=2000,
        n_delay_bins=30,
        noise_level=.5,
        signal_profile='exponential',
        profile_params={'amplitude': 0.0}  # No signal
    )
    
    # Run analysis
    task = PumpProbeAnalysis(config)
    input_data = PumpProbeAnalysisInput(
        config=config,
        load_data_output=load_data_output,
        masks_output=masks_output
    )
    
    output = task.run(input_data)
    
    # Check p-value distribution under null hypothesis
    p_values = output.p_values
    # Should be roughly uniform under null hypothesis
    _, p_uniform = stats.kstest(p_values, 'uniform')
    assert p_uniform > 0.05, "P-values not uniform under null hypothesis"
    
    # Check false positive rate
    fpr = np.mean(p_values < config['pump_probe_analysis']['significance_level'])
    assert np.abs(fpr - config['pump_probe_analysis']['significance_level']) < 0.02, \
        f"False positive rate {fpr:.3f} differs from nominal {config['pump_probe_analysis']['significance_level']}"

def test_robustness():
    """Test robustness to various data quality issues."""
    config = {
        'pump_probe_analysis': {
            'min_count': 10,
            'significance_level': 0.05
        }
    }
    
    # Test with high noise
    load_data_output, masks_output, _ = generate_synthetic_pump_probe_data(
        n_frames=2000,
        noise_level=0.2,  # High noise
        signal_profile='exponential',
        profile_params={'amplitude': 0.5, 'decay_time': 5.0}
    )
    
    task = PumpProbeAnalysis(config)
    input_data = PumpProbeAnalysisInput(
        config=config,
        load_data_output=load_data_output,
        masks_output=masks_output
    )
    
    # Should run without errors
    output = task.run(input_data)
    
    # Test with uneven frame distribution
    load_data_output, masks_output, _ = generate_synthetic_pump_probe_data(
        n_frames=1000,  # Fewer frames
        n_delay_bins=30,
        min_frames_per_bin=5  # Below minimum
    )
    
    # Should raise ValueError due to insufficient frames
    with pytest.raises(ValueError, match="No delay groups met the minimum frame count requirement"):
        task.run(PumpProbeAnalysisInput(
            config=config,
            load_data_output=load_data_output,
            masks_output=masks_output
        ))

if __name__ == '__main__':
    # Run time-dependent signal tests with visualization
    print("Running pump-probe analysis tests...")
    test_pump_probe_time_dependent()
    print("\nTests complete. Check the diagnostic_plots directory for visualizations.")
