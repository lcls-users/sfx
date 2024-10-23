import pytest
from pathlib import Path
import numpy as np
import tempfile

from btx.processing.tasks.load_data import LoadData
from btx.processing.types import LoadDataInput
from btx.processing.tests.functional.data_generators import generate_synthetic_frames

def test_load_data_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        'setup': {
            'run': 123,
            'exp': 'test',
        },
        'load_data': {
            'roi': [0, 100, 0, 100],
            'time_bin': 2.0
        }
    }
    
    task = LoadData(valid_config)  # Should not raise
    
    # Test missing setup section
    invalid_config = valid_config.copy()
    del invalid_config['setup']
    with pytest.raises(ValueError, match="Missing 'setup' section"):
        LoadData(invalid_config)
    
    # Test invalid ROI
    invalid_config = valid_config.copy()
    invalid_config['load_data']['roi'] = [100, 0, 0, 100]  # Start > end
    with pytest.raises(ValueError, match="Invalid ROI coordinates"):
        LoadData(invalid_config)

def test_load_data_synthetic():
    """Test LoadData with synthetic data."""
    # Generate synthetic data
    num_frames = 1000
    rows = cols = 100
    data, I0, delays, on_mask, off_mask = generate_synthetic_frames(
        num_frames, rows, cols
    )
    
    # Create config
    config = {
        'setup': {
            'run': 123,
            'exp': 'test',
        },
        'load_data': {
            'roi': [0, rows, 0, cols],
            'time_bin': 2.0
        }
    }
    
    # Create input
    input_data = LoadDataInput(
        config=config,
        data=data,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=on_mask,
        laser_off_mask=off_mask
    )
    
    # Run task
    task = LoadData(config)
    output = task.run(input_data)
    
    # Validate output shapes
    assert output.data.shape == (num_frames, rows, cols)
    assert output.I0.shape == (num_frames,)
    assert output.laser_delays.shape == (num_frames,)
    assert output.binned_delays.shape == (num_frames,)
    assert output.laser_on_mask.shape == (num_frames,)
    assert output.laser_off_mask.shape == (num_frames,)
    
    # Validate binned delays
    bin_size = config['load_data']['time_bin']
    diffs = np.diff(np.unique(output.binned_delays))
    assert np.allclose(diffs, bin_size, atol=1e-10)

def test_load_data_visual():
    """Generate visual diagnostic plots for manual inspection."""
    # Save plots in a fixed location under the project's processing directory
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'load_data'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visual test plots in: {save_dir}")
    
    # ... (rest of the test remains the same, but use save_dir instead of temp_dir)
    
    # Generate synthetic data
    num_frames = 1000
    rows = cols = 100
    data, I0, delays, on_mask, off_mask = generate_synthetic_frames(
        num_frames, rows, cols
    )
    
    # Create config
    config = {
        'setup': {
            'run': 123,
            'exp': 'test',
        },
        'load_data': {
            'roi': [0, rows, 0, cols],
            'time_bin': 2.0
        }
    }
    
    # Create input
    input_data = LoadDataInput(
        config=config,
        data=data,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=on_mask,
        laser_off_mask=off_mask
    )
    
    # Run task
    task = LoadData(config)
    output = task.run(input_data)
    
    # Generate diagnostic plots
    task.plot_diagnostics(output, save_dir)
    
    # Verify plot was created
    plot_file = save_dir / 'load_data_diagnostics.png'
    assert plot_file.exists(), f"Diagnostic plot not created at {plot_file}"
    
    print(f"Generated diagnostic plot: {plot_file}")
    return save_dir  # Return path for manual inspection

if __name__ == '__main__':
    # Run visual test and print location of plots
    plot_dir = test_load_data_visual()
    print(f"\nVisual test complete. Inspect plots at: {plot_dir}")
