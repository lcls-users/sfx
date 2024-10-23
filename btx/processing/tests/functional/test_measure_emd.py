import pytest
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from btx.processing.tasks.measure_emd import MeasureEMD
from btx.processing.types import (
    MeasureEMDInput,
    MakeHistogramOutput,
    LoadDataOutput
)

def generate_synthetic_histograms(
    num_bins: int = 50,
    rows: int = 100,
    cols: int = 100,
    background_mean: float = 10.0,
    signal_mean: float = 15.0,
    noise_level: float = 1.0,
    roi_x: Tuple[int, int] = (20, 40),
    roi_y: Tuple[int, int] = (20, 40)
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic histograms with known signal and background regions."""
    # Create background histograms
    histograms = np.random.normal(
        background_mean,
        noise_level,
        (num_bins, rows, cols)
    )
    
    # Add signal region (different distribution)
    signal_mask = np.zeros((rows, cols), dtype=bool)
    signal_mask[60:80, 60:80] = True
    
    for i in range(num_bins):
        histograms[i, signal_mask] = np.random.normal(
            signal_mean,
            noise_level,
            size=np.sum(signal_mask)
        )
    
    # Ensure histograms are positive
    histograms = np.abs(histograms)
    
    # Create bin edges
    bin_edges = np.linspace(0, 30, num_bins + 1)
    
    return histograms, bin_edges

def test_measure_emd_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        },
        'calculate_emd': {
            'num_permutations': 100
        }
    }
    
    task = MeasureEMD(valid_config)  # Should not raise
    
    # Test missing section
    invalid_config = {}
    with pytest.raises(ValueError, match="Missing 'setup' section"):
        MeasureEMD(invalid_config)
    
    # Test missing ROI
    invalid_config = {'setup': {}}
    with pytest.raises(ValueError, match="Missing background_roi_coords"):
        MeasureEMD(invalid_config)
    
    # Test invalid ROI format
    invalid_config = {
        'setup': {
            'background_roi_coords': [0, 1, 2]  # Wrong length
        }
    }
    with pytest.raises(ValueError, match="must be \\[x1, x2, y1, y2\\]"):
        MeasureEMD(invalid_config)
        
    # Test invalid ROI coordinates
    invalid_config = {
        'setup': {
            'background_roi_coords': [40, 20, 20, 40]  # x2 < x1
        }
    }
    with pytest.raises(ValueError, match="Invalid ROI coordinates"):
        MeasureEMD(invalid_config)

def test_measure_emd_synthetic():
    """Test MeasureEMD with synthetic data."""
    # Generate synthetic data
    num_bins = 50
    rows = cols = 100
    histograms, bin_edges = generate_synthetic_histograms(
        num_bins=num_bins,
        rows=rows,
        cols=cols
    )
    
    # Create histogram output
    histogram_output = MakeHistogramOutput(
        histograms=histograms,
        bin_edges=bin_edges,
        bin_centers=(bin_edges[:-1] + bin_edges[1:]) / 2
    )
    
    # Create config
    config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        },
        'calculate_emd': {
            'num_permutations': 100
        }
    }
    
    # Create input
    input_data = MeasureEMDInput(
        config=config,
        histogram_output=histogram_output
    )
    
    # Run task
    task = MeasureEMD(config)
    output = task.run(input_data)
    
    # Validate output shapes
    assert output.emd_values.shape == (rows, cols)
    assert len(output.null_distribution) == config['calculate_emd']['num_permutations']
    assert len(output.avg_histogram) == num_bins
    
    # Validate EMD values
    assert np.all(output.emd_values >= 0)  # EMD should be non-negative
    assert np.all(np.isfinite(output.emd_values))  # No inf/nan values
    
    # Check that signal region has higher EMD values
    signal_region = output.emd_values[60:80, 60:80]
    background_region = output.emd_values[20:40, 20:40]
    assert np.mean(signal_region) > np.mean(background_region)
    
    # Validate null distribution
    assert np.all(output.null_distribution >= 0)
    assert np.all(np.isfinite(output.null_distribution))
    
    # Check average histogram
    assert np.all(output.avg_histogram >= 0)
    assert np.all(np.isfinite(output.avg_histogram))

def test_measure_emd_edge_cases():
    """Test MeasureEMD with edge cases."""
    # Generate basic synthetic data
    num_bins = 50
    rows = cols = 100
    histograms, bin_edges = generate_synthetic_histograms(
        num_bins=num_bins,
        rows=rows,
        cols=cols
    )
    
    histogram_output = MakeHistogramOutput(
        histograms=histograms,
        bin_edges=bin_edges,
        bin_centers=(bin_edges[:-1] + bin_edges[1:]) / 2
    )
    
    # Test case 1: ROI at image boundary
    config = {
        'setup': {
            'background_roi_coords': [0, 20, 0, 20]
        }
    }
    
    task = MeasureEMD(config)
    input_data = MeasureEMDInput(config=config, histogram_output=histogram_output)
    output = task.run(input_data)  # Should not raise
    
    # Test case 2: Small ROI (should warn)
    config = {
        'setup': {
            'background_roi_coords': [0, 5, 0, 5]
        }
    }
    
    task = MeasureEMD(config)
    input_data = MeasureEMDInput(config=config, histogram_output=histogram_output)
    with pytest.warns(RuntimeWarning, match="Background ROI contains only"):
        output = task.run(input_data)
    
    # Test case 3: ROI outside image bounds
    config = {
        'setup': {
            'background_roi_coords': [0, 20, 90, 110]
        }
    }
    
    task = MeasureEMD(config)
    input_data = MeasureEMDInput(config=config, histogram_output=histogram_output)
    with pytest.raises(ValueError, match="Background ROI .* invalid for histograms"):
        output = task.run(input_data)
    
    # Test case 4: Empty histograms
    empty_histograms = np.zeros_like(histograms)
    empty_output = MakeHistogramOutput(
        histograms=empty_histograms,
        bin_edges=bin_edges,
        bin_centers=(bin_edges[:-1] + bin_edges[1:]) / 2
    )
    
    config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        }
    }
    
    task = MeasureEMD(config)
    input_data = MeasureEMDInput(config=config, histogram_output=empty_output)
    with pytest.raises(ValueError, match="Background ROI contains no data"):
        output = task.run(input_data)

def test_measure_emd_visual():
    """Generate visual diagnostic plots for manual inspection."""
    # Set up output directory
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'measure_emd'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visual test plots in: {save_dir}")
    
    # Generate synthetic data with clear signal
    num_bins = 50
    rows = cols = 100
    histograms, bin_edges = generate_synthetic_histograms(
        num_bins=num_bins,
        rows=rows,
        cols=cols,
        background_mean=10.0,
        signal_mean=15.0,
        noise_level=0.5  # Lower noise for clearer visualization
    )
    
    histogram_output = MakeHistogramOutput(
        histograms=histograms,
        bin_edges=bin_edges,
        bin_centers=(bin_edges[:-1] + bin_edges[1:]) / 2
    )
    
    # Create config
    config = {
        'setup': {
            'background_roi_coords': [20, 40, 20, 40]
        },
        'calculate_emd': {
            'num_permutations': 1000  # More permutations for better visualization
        }
    }
    
    # Create input
    input_data = MeasureEMDInput(
        config=config,
        histogram_output=histogram_output
    )
    
    # Run task
    task = MeasureEMD(config)
    output = task.run(input_data)
    
    # Generate diagnostic plots
    task.plot_diagnostics(output, save_dir)
    
    # Verify plots were created
    expected_plots = [
        'measure_emd_diagnostics.png',
        'measure_emd_distribution.png'
    ]
    
    for plot_name in expected_plots:
        plot_file = save_dir / plot_name
        assert plot_file.exists(), f"Diagnostic plot not created at {plot_file}"
        print(f"Generated diagnostic plot: {plot_file}")
    
    plt.close('all')  # Clean up

if __name__ == '__main__':
    # Run visual test to generate plots
    test_measure_emd_visual()
    print("\nVisual test complete. Check the diagnostic plots in the output directory.")
