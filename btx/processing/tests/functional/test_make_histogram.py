import pytest
from pathlib import Path
import numpy as np

from btx.processing.tasks.make_histogram import MakeHistogram
from btx.processing.btx_types import MakeHistogramInput, LoadDataOutput
from btx.processing.tests.functional.data_generators import generate_synthetic_frames

def test_make_histogram_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        'make_histogram': {
            'bin_boundaries': np.arange(5, 30, 0.2),
            'hist_start_bin': 1
        }
    }
    
    task = MakeHistogram(valid_config)  # Should not raise
    
    # Test missing section
    invalid_config = {}
    with pytest.raises(ValueError, match="Missing 'make_histogram' section"):
        MakeHistogram(invalid_config)
    
    # Test invalid hist_start_bin
    invalid_config = {
        'make_histogram': {
            'bin_boundaries': np.arange(5, 30, 0.2),
            'hist_start_bin': -1  # Invalid negative index
        }
    }
    with pytest.raises(ValueError, match="hist_start_bin must be between"):
        MakeHistogram(invalid_config)

def test_make_histogram_synthetic():
    """Test MakeHistogram with synthetic data."""
    # Generate synthetic data
    num_frames = 1000
    rows = cols = 100
    data, I0, delays, on_mask, off_mask = generate_synthetic_frames(
        num_frames, rows, cols
    )
    
    # Scale data to match expected range
    data = 5 + (25 * (data - data.min()) / (data.max() - data.min()))
    
    # Create LoadDataOutput
    load_data_output = LoadDataOutput(
        data=data,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=on_mask,
        laser_off_mask=off_mask,
        binned_delays=delays
    )
    
    # Create config
    config = {
        'make_histogram': {
            'bin_boundaries': np.arange(5, 30, 0.2),
            'hist_start_bin': 1
        }
    }
    
    # Create input
    input_data = MakeHistogramInput(
        config=config,
        load_data_output=load_data_output
    )
    
    # Run task
    task = MakeHistogram(config)
    output = task.run(input_data)
    
    # Validate output shapes
    n_bins = len(config['make_histogram']['bin_boundaries']) - 1
    expected_bins = n_bins - config['make_histogram']['hist_start_bin']
    assert output.histograms.shape == (expected_bins, rows, cols)
    
    # Validate histogram properties
    assert np.all(output.histograms >= 1e-9)  # Check small constant was added
    assert np.all(np.isfinite(output.histograms))  # Check for any inf/nan values
    
    # Validate bin structure
    assert len(output.bin_edges) == expected_bins 
    assert len(output.bin_centers) == expected_bins
#    assert np.allclose(
#        output.bin_centers,
#        (output.bin_edges[:-1] + output.bin_edges[1:]) / 2
#    )

def test_make_histogram_visual():
    """Generate visual diagnostic plots for manual inspection."""
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'make_histogram'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visual test plots in: {save_dir}")
    
    # Generate synthetic data
    num_frames = 1000
    rows = cols = 100
    data, I0, delays, on_mask, off_mask = generate_synthetic_frames(
        num_frames, rows, cols
    )
    
    # Scale data to match expected range
    data = 5 + (25 * (data - data.min()) / (data.max() - data.min()))
    
    # Create LoadDataOutput
    load_data_output = LoadDataOutput(
        data=data,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=on_mask,
        laser_off_mask=off_mask,
        binned_delays=delays
    )
    
    # Create config
    config = {
        'make_histogram': {
            'bin_boundaries': np.arange(5, 30, 0.2),
            'hist_start_bin': 1
        }
    }
    
    # Create input
    input_data = MakeHistogramInput(
        config=config,
        load_data_output=load_data_output
    )
    
    # Run task
    task = MakeHistogram(config)
    output = task.run(input_data)
    
    # Generate diagnostic plots
    task.plot_diagnostics(output, save_dir)
    
    # Verify plots were created
    for plot_name in ['make_histogram_diagnostics.png', 'make_histogram_central_pixels.png']:
        plot_file = save_dir / plot_name
        assert plot_file.exists(), f"Diagnostic plot not created at {plot_file}"
        print(f"Generated diagnostic plot: {plot_file}")
    
    return save_dir

if __name__ == '__main__':
    # Run visual test and print location of plots
    plot_dir = test_make_histogram_visual()
    print(f"\nVisual test complete. Inspect plots at: {plot_dir}")
