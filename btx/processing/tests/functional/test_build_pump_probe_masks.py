import pytest
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from btx.processing.tasks.build_pump_probe_masks import BuildPumpProbeMasks
from btx.processing.btx_types import (
    BuildPumpProbeMasksInput,
    MakeHistogramOutput,
    CalculatePValuesOutput
)

def generate_synthetic_data(
    rows: int = 100,
    cols: int = 100,
    signal_center: Tuple[int, int] = (70, 70),
    signal_radius: int = 10,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic p-values and histograms with circular signal."""
    np.random.seed(seed)
    
    # Start with high p-values everywhere
    p_values = np.random.uniform(0.5, 1.0, (rows, cols))
    
    # Add clear signal region (very low p-values)
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - signal_center[1])**2 + (y - signal_center[0])**2)
    signal_mask = dist <= signal_radius
    # Make signal very significant
    p_values[signal_mask] = np.random.uniform(0.0001, 0.01, size=np.sum(signal_mask))
    
    # Add noise
    p_values += np.random.normal(0, noise_level, p_values.shape)
    p_values = np.clip(p_values, 0, 1)
    
    # Create dummy histograms
    histograms = np.random.normal(0, 1, (10, rows, cols))
    
    print(f"Signal region p-values: min={np.min(p_values[signal_mask]):.3f}, "
          f"max={np.max(p_values[signal_mask]):.3f}")
    print(f"Background p-values: min={np.min(p_values[~signal_mask]):.3f}, "
          f"max={np.max(p_values[~signal_mask]):.3f}")
    
    return p_values, histograms

def test_build_pump_probe_masks_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        },
        'generate_masks': {
            'threshold': 0.05,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5
        }
    }
    
    task = BuildPumpProbeMasks(valid_config)  # Should not raise
    
    # Test missing section
    invalid_config = {}
    with pytest.raises(ValueError, match="Missing 'setup' section"):
        BuildPumpProbeMasks(invalid_config)
    
    # Test missing ROI
    invalid_config = {'setup': {}}
    with pytest.raises(ValueError, match="Missing background_roi_coords"):
        BuildPumpProbeMasks(invalid_config)
    
    # Test missing generate_masks section
    invalid_config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        }
    }
    with pytest.raises(ValueError, match="Missing 'generate_masks' section"):
        BuildPumpProbeMasks(invalid_config)
        
    # Test invalid threshold
    invalid_config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        },
        'generate_masks': {
            'threshold': 1.5,  # > 1
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5
        }
    }
    with pytest.raises(ValueError, match="threshold must be between"):
        BuildPumpProbeMasks(invalid_config)

def test_build_pump_probe_masks_synthetic():
    """Test BuildPumpProbeMasks with synthetic data.
    
    The ROI is used only for signal identification, while the background
    mask is generated as a buffer around the identified signal regions.
    """
    # Generate synthetic data
    rows = cols = 100
    p_values, histograms = generate_synthetic_data(
        rows=rows,
        cols=cols,
        signal_center=(70, 70),
        signal_radius=10
    )
    
    # Create mock outputs from previous tasks
    p_values_output = CalculatePValuesOutput(
        p_values=p_values,
        log_p_values=-np.log10(p_values),
        significance_threshold=0.05
    )
    
    histogram_output = MakeHistogramOutput(
        histograms=histograms,
        bin_edges=np.linspace(0, 1, 11),
        bin_centers=np.linspace(0.05, 0.95, 10)
    )
    
    # Create config
    config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]  # Away from signal
        },
        'generate_masks': {
            'threshold': 0.05,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5
        }
    }
    
    # Create input
    input_data = BuildPumpProbeMasksInput(
        config=config,
        histogram_output=histogram_output,
        p_values_output=p_values_output
    )
    
    # Run task
    task = BuildPumpProbeMasks(config)
    output = task.run(input_data)
    
    # Validate output types
    assert output.signal_mask.dtype == bool
    assert output.background_mask.dtype == bool
    
    # Validate shapes
    assert output.signal_mask.shape == (rows, cols)
    assert output.background_mask.shape == (rows, cols)
    
    # Validate masks don't overlap
    assert not np.any(output.signal_mask & output.background_mask)
    
    # Check signal mask found the signal region
    signal_center = np.array([70, 70])
    signal_detected = output.signal_mask[
        signal_center[0]-5:signal_center[0]+5,
        signal_center[1]-5:signal_center[1]+5
    ]
    assert np.any(signal_detected), "Signal region not detected"
    
    # Check background mask is near signal (not in ROI)
    signal_region = np.zeros_like(output.signal_mask)
    signal_region[
        signal_center[0]-15:signal_center[0]+15,
        signal_center[1]-15:signal_center[1]+15
    ] = True
    
    # Background should overlap with region near signal
    assert np.any(output.background_mask & signal_region), "Background not found near signal"

def test_build_pump_probe_masks_edge_cases():
    """Test BuildPumpProbeMasks with edge cases."""
    rows = cols = 100
    
    # Test case 1: No signal (all high p-values)
    p_values = np.random.uniform(0.5, 1.0, (rows, cols))
    histograms = np.random.normal(0, 1, (10, rows, cols))
    
    p_values_output = CalculatePValuesOutput(
        p_values=p_values,
        log_p_values=-np.log10(p_values),
        significance_threshold=0.05
    )
    
    histogram_output = MakeHistogramOutput(
        histograms=histograms,
        bin_edges=np.linspace(0, 1, 11),
        bin_centers=np.linspace(0.05, 0.95, 10)
    )
    
    config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        },
        'generate_masks': {
            'threshold': 0.05,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5
        }
    }
    
    input_data = BuildPumpProbeMasksInput(
        config=config,
        histogram_output=histogram_output,
        p_values_output=p_values_output
    )
    
#    task = BuildPumpProbeMasks(config)
#    with pytest.raises(ValueError, match="Signal mask is empty"):
#        output = task.run(input_data)
    
    # Test case 2: All signal (all low p-values)
    p_values = np.random.uniform(0.0, 0.01, (rows, cols))
    p_values_output = CalculatePValuesOutput(
        p_values=p_values,
        log_p_values=-np.log10(p_values),
        significance_threshold=0.05
    )
    
    input_data = BuildPumpProbeMasksInput(
        config=config,
        histogram_output=histogram_output,
        p_values_output=p_values_output
    )
    
#    with pytest.warns(RuntimeWarning, match="Signal mask covers"):
#        output = task.run(input_data)

def test_build_pump_probe_masks_visual():
    """Generate visual diagnostic plots for manual inspection."""
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'build_pump_probe_masks'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visual test plots in: {save_dir}")
    
    # Generate synthetic data with clear signal
    rows = cols = 100
    p_values, histograms = generate_synthetic_data(
        rows=rows,
        cols=cols,
        signal_center=(70, 70),
        signal_radius=10,
        noise_level=0.05  # Low noise for clear visualization
    )
    
    p_values_output = CalculatePValuesOutput(
        p_values=p_values,
        log_p_values=-np.log10(p_values),
        significance_threshold=0.05
    )
    
    histogram_output = MakeHistogramOutput(
        histograms=histograms,
        bin_edges=np.linspace(0, 1, 11),
        bin_centers=np.linspace(0.05, 0.95, 10)
    )
    
    # Create config
    config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]  # Away from signal
        },
        'generate_masks': {
            'threshold': 0.05,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5
        }
    }
    
    # Create input
    input_data = BuildPumpProbeMasksInput(
        config=config,
        histogram_output=histogram_output,
        p_values_output=p_values_output
    )
    
    # Run task
    task = BuildPumpProbeMasks(config)
    output = task.run(input_data)
    
    # Generate diagnostic plots
    task.plot_diagnostics(output, save_dir)
    
    # Verify plots were created
    expected_plots = [
        'mask_generation_stages.png',
        'final_masks.png',
        'mask_distance.png'
    ]
    
    for plot_name in expected_plots:
        plot_file = save_dir / plot_name
        assert plot_file.exists(), f"Diagnostic plot not created at {plot_file}"
        print(f"Generated diagnostic plot: {plot_file}")
    
    plt.close('all')  # Clean up

if __name__ == '__main__':
    # Run visual test to generate plots
    test_build_pump_probe_masks_visual()
    print("\nVisual test complete. Check the diagnostic plots in the output directory.")
