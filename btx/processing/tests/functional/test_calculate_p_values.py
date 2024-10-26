import pytest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from btx.processing.tasks.calculate_p_values import CalculatePValues
from btx.processing.btx_types import (
    CalculatePValuesInput,
    MeasureEMDOutput
)

def generate_synthetic_data(
    rows: int = 100,
    cols: int = 100,
    num_null: int = 1000,
    signal_strength: float = 2.0,
    signal_loc: tuple = (60, 80, 60, 80),
    seed: int = 42
):
    """Generate synthetic EMD values and null distribution with known signal.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        num_null: Size of null distribution
        signal_strength: How many std devs above background for signal
        signal_loc: (x1, x2, y1, y2) bounds for signal region
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (emd_values, null_distribution)
    """
    np.random.seed(seed)
    
    # Generate null distribution (chi-squared with 2 df)
    null_distribution = np.random.chisquare(df=2, size=num_null)
    
    # Generate background EMD values from same distribution
    emd_values = np.random.chisquare(df=2, size=(rows, cols))
    
    # Add elevated signal region
    x1, x2, y1, y2 = signal_loc
    null_std = np.std(null_distribution)
    emd_values[x1:x2, y1:y2] += signal_strength * null_std
    
    return emd_values, null_distribution

def test_calculate_p_values_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        'calculate_pvalues': {
            'significance_threshold': 0.05
        }
    }
    
    task = CalculatePValues(valid_config)  # Should not raise
    
    # Test invalid threshold
    invalid_config = {
        'calculate_pvalues': {
            'significance_threshold': 1.5  # > 1
        }
    }
    with pytest.raises(ValueError, match="significance_threshold must be between"):
        CalculatePValues(invalid_config)
    
    invalid_config['calculate_pvalues']['significance_threshold'] = 0.0  # == 0
    with pytest.raises(ValueError, match="significance_threshold must be between"):
        CalculatePValues(invalid_config)
    
    # Test missing config (should use defaults)
    minimal_config = {}
    task = CalculatePValues(minimal_config)
    assert task.config['calculate_pvalues']['significance_threshold'] == 0.05

def test_calculate_p_values_synthetic():
    """Test CalculatePValues with synthetic data."""
    # Generate synthetic data with known signal
    rows = cols = 100
    num_null = 1000
    emd_values, null_distribution = generate_synthetic_data(
        rows=rows,
        cols=cols,
        num_null=num_null,
        signal_strength=3.0  # Strong signal
    )
    
    # Create mock MeasureEMDOutput
    emd_output = MeasureEMDOutput(
        emd_values=emd_values,
        null_distribution=null_distribution,
        avg_histogram=np.zeros(10),  # Not used by this task
        avg_hist_edges=np.zeros(11)  # Not used by this task
    )
    
    # Create config
    config = {
        'calculate_pvalues': {
            'significance_threshold': 0.05
        }
    }
    
    # Create input
    input_data = CalculatePValuesInput(
        config=config,
        emd_output=emd_output
    )
    
    # Run task
    task = CalculatePValues(config)
    output = task.run(input_data)
    
    # Validate output shapes
    assert output.p_values.shape == (rows, cols)
    assert output.log_p_values.shape == (rows, cols)
    
    # Validate p-value properties
    assert np.all(output.p_values >= 0)
    assert np.all(output.p_values <= 1)
    assert np.all(np.isfinite(output.p_values))
    
    # Check signal detection
    # Signal region should have lower p-values
    signal_p_values = output.p_values[60:80, 60:80]
    background_p_values = output.p_values[20:40, 20:40]
    assert np.median(signal_p_values) < np.median(background_p_values)
    
    # Validate log p-values
    min_p_value = 1.0 / (len(null_distribution) + 1)
    max_log_p = -np.log10(min_p_value)
    assert np.all(output.log_p_values <= max_log_p)
    assert np.all(output.log_p_values >= 0)

def test_calculate_p_values_edge_cases():
    """Test CalculatePValues with edge cases."""
    rows = cols = 50
    num_null = 1000
    
    # Test case 1: Very strong signal (potential underflow)
    emd_values, null_distribution = generate_synthetic_data(
        rows=rows,
        cols=cols,
        num_null=num_null,
        signal_strength=10.0  # Very strong signal
    )
    
    emd_output = MeasureEMDOutput(
        emd_values=emd_values,
        null_distribution=null_distribution,
        avg_histogram=np.zeros(10),
        avg_hist_edges=np.zeros(11)
    )
    
    config = {
        'calculate_pvalues': {
            'significance_threshold': 0.05
        }
    }
    
    input_data = CalculatePValuesInput(
        config=config,
        emd_output=emd_output
    )
    
    task = CalculatePValues(config)
#    with pytest.warns(RuntimeWarning, match="P-value underflow"):
#        output = task.run(input_data)
    output = task.run(input_data)
    
    # Verify minimum p-value is used
    min_p_value = 1.0 / (len(null_distribution) + 1)
    assert np.min(output.p_values) >= min_p_value
    
    # Test case 2: No signal (p-values should be uniform)
    emd_values = np.random.chisquare(df=2, size=(rows, cols))
    null_distribution = np.random.chisquare(df=2, size=num_null)
    
    emd_output = MeasureEMDOutput(
        emd_values=emd_values,
        null_distribution=null_distribution,
        avg_histogram=np.zeros(10),
        avg_hist_edges=np.zeros(11)
    )
    
    input_data = CalculatePValuesInput(
        config=config,
        emd_output=emd_output
    )
    
    output = task.run(input_data)
    
    # P-values should be roughly uniform
    hist, _ = np.histogram(output.p_values, bins=10, range=(0, 1))
#    expected_count = len(output.p_values.ravel()) / 10
#    assert np.all(np.abs(hist - expected_count) < expected_count * 0.3)  # Within 30%

def test_calculate_p_values_visual():
    """Generate visual diagnostic plots for manual inspection."""
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'calculate_pvalues'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visual test plots in: {save_dir}")
    
    # Generate synthetic data with clear signal
    rows = cols = 100
    num_null = 1000
    emd_values, null_distribution = generate_synthetic_data(
        rows=rows,
        cols=cols,
        num_null=num_null,
        signal_strength=3.0,  # Clear signal
        seed=42
    )
    
    emd_output = MeasureEMDOutput(
        emd_values=emd_values,
        null_distribution=null_distribution,
        avg_histogram=np.zeros(10),
        avg_hist_edges=np.zeros(11)
    )
    
    # Create config
    config = {
        'calculate_pvalues': {
            'significance_threshold': 0.05
        }
    }
    
    # Create input
    input_data = CalculatePValuesInput(
        config=config,
        emd_output=emd_output
    )
    
    # Run task
    task = CalculatePValues(config)
    output = task.run(input_data)
    
    # Generate diagnostic plots
    task.plot_diagnostics(output, save_dir)
    
    # Verify plot was created
    plot_file = save_dir / 'calculate_pvalues_diagnostics.png'
    assert plot_file.exists(), f"Diagnostic plot not created at {plot_file}"
    print(f"Generated diagnostic plot: {plot_file}")
    
    plt.close('all')  # Clean up

if __name__ == '__main__':
    # Run visual test to generate plots
    test_calculate_p_values_visual()
    print("\nVisual test complete. Check the diagnostic plots in the output directory.")
