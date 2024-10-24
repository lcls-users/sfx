# btx/processing/tests/functional/test_pipeline_integration.py
from typing import Dict, Any
import pytest
import numpy as np
from pathlib import Path

from btx.processing.core import (
    PipelineBuilder,
    task_registry,
    TaskAdapter,
    Pipeline
)
from btx.processing.btx_types import (
    LoadDataInput, LoadDataOutput,
    MakeHistogramInput, MakeHistogramOutput,
    MeasureEMDInput, MeasureEMDOutput,
    CalculatePValuesInput, CalculatePValuesOutput,
    BuildPumpProbeMasksInput, BuildPumpProbeMasksOutput,
    PumpProbeAnalysisInput, PumpProbeAnalysisOutput
)
from btx.processing.tasks import (
    LoadData, MakeHistogram, MeasureEMD,
    CalculatePValues, BuildPumpProbeMasks, PumpProbeAnalysis
)

from btx.processing.tests.functional.data_generators import generate_synthetic_frames

@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Fixture providing minimal valid configuration for all tasks."""
    return {
        'setup': {
            'run': 123,
            'exp': 'test_exp',
            'background_roi_coords': [20, 40, 20, 40],  # For EMD and masks
            'output_dir': 'results'
        },
        'load_data': {
            'roi': [0, 100, 0, 100],  # Assuming 100x100 frames
            'time_bin': 2.0,  # 2ps bins
            'energy_filter': [8.8, 5],
            'i0_threshold': 200
        },
        'make_histogram': {
            'bin_boundaries': np.arange(5, 30, 0.2),
            'hist_start_bin': 1
        },
        'calculate_emd': {
            'num_permutations': 100  # Reduced for testing
        },
        'calculate_pvalues': {
            'significance_threshold': 0.05
        },
        'generate_masks': {
            'threshold': 0.05,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5
        },
        'pump_probe_analysis': {
            'min_count': 10,
            'significance_level': 0.05
        }
    }

@pytest.fixture
def pipeline_diagnostics_dir(tmp_path: Path) -> Path:
    """Fixture providing temporary directory for pipeline diagnostics."""
    diag_dir = tmp_path / "pipeline_diagnostics"
    diag_dir.mkdir()
    return diag_dir

@pytest.fixture
def synthetic_data():
    """Fixture providing synthetic XPP data."""
    # Generate synthetic frames with base pattern + signal
    num_frames = 1000
    rows = cols = 100
    return generate_synthetic_frames(
        num_frames=num_frames,
        rows=rows,
        cols=cols,
        seed=42
    )

def test_histogram_pipeline(
    base_config: Dict[str, Any],
    pipeline_diagnostics_dir: Path,
    synthetic_data: tuple
):
    """Test pipeline from LoadData through MeasureEMD."""
    data, I0, delays, on_mask, off_mask = synthetic_data
    
    # Register tasks
    task_registry.register(
        "load_data", LoadData, LoadDataInput, LoadDataOutput
    )
    task_registry.register(
        "make_histogram", MakeHistogram, MakeHistogramInput, MakeHistogramOutput
    )
    task_registry.register(
        "measure_emd", MeasureEMD, MeasureEMDInput, MeasureEMDOutput
    )
    
    # Build pipeline
    pipeline = (PipelineBuilder("Histogram Pipeline")
        .add("load_data", task_registry.create("load_data", base_config))
        .add("make_histogram", task_registry.create("make_histogram", base_config),
             ["load_data"])
        .add("measure_emd", task_registry.create("measure_emd", base_config),
             ["make_histogram"])
        .set_diagnostics_dir(pipeline_diagnostics_dir)
        .build())
    
    # Create input with synthetic data
    input_data = LoadDataInput(
        config=base_config,
        data=data,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=on_mask,
        laser_off_mask=off_mask
    )
    
    # Run pipeline
    results = pipeline.run(input_data)
    
    # Validate results
    assert results.success
    assert set(results.execution_order) == {"load_data", "make_histogram", "measure_emd"}
    
    # Check LoadData output
    load_result = results.results["load_data"].output
    assert isinstance(load_result, LoadDataOutput)
    assert load_result.data.shape == data.shape
    assert load_result.binned_delays.shape == delays.shape
    
    # Check MakeHistogram output
    hist_result = results.results["make_histogram"].output
    assert isinstance(hist_result, MakeHistogramOutput)
    n_bins = len(base_config['make_histogram']['bin_boundaries']) - 1
    expected_bins = n_bins - base_config['make_histogram']['hist_start_bin']
    assert hist_result.histograms.shape == (expected_bins, rows, cols)
    
    # Check MeasureEMD output
    emd_result = results.results["measure_emd"].output
    assert isinstance(emd_result, MeasureEMDOutput)
    assert emd_result.emd_values.shape == (rows, cols)
    assert len(emd_result.null_distribution) == base_config['calculate_emd']['num_permutations']

def test_analysis_pipeline(
    base_config: Dict[str, Any],
    pipeline_diagnostics_dir: Path,
    synthetic_data: tuple
):
    """Test pipeline from MeasureEMD through BuildPumpProbeMasks."""
    # Mock a MeasureEMDOutput as input
    rows = cols = 100
    mock_emd_output = MeasureEMDOutput(
        emd_values=np.random.normal(0, 1, (rows, cols)),
        null_distribution=np.abs(np.random.normal(0, 1, 1000)),
        avg_histogram=np.ones(50),  # Dummy histogram
        avg_hist_edges=np.arange(51)
    )
    
    # Register tasks
    task_registry.register(
        "calculate_pvalues", CalculatePValues,
        CalculatePValuesInput, CalculatePValuesOutput
    )
    task_registry.register(
        "build_masks", BuildPumpProbeMasks,
        BuildPumpProbeMasksInput, BuildPumpProbeMasksOutput
    )
    
    # Build pipeline
    pipeline = (PipelineBuilder("Analysis Pipeline")
        .add("calculate_pvalues",
             task_registry.create("calculate_pvalues", base_config))
        .add("build_masks",
             task_registry.create("build_masks", base_config),
             ["calculate_pvalues"])
        .set_diagnostics_dir(pipeline_diagnostics_dir)
        .build())
    
    # Create input
    input_data = CalculatePValuesInput(
        config=base_config,
        emd_output=mock_emd_output
    )
    
    # Run pipeline
    results = pipeline.run(input_data)
    
    # Validate results
    assert results.success
    assert set(results.execution_order) == {"calculate_pvalues", "build_masks"}
    
    # Check CalculatePValues output
    pval_result = results.results["calculate_pvalues"].output
    assert isinstance(pval_result, CalculatePValuesOutput)
    np.testing.assert_array_equal(pval_result.p_values.shape, (rows, cols))
    np.testing.assert_array_equal(pval_result.log_p_values.shape, (rows, cols))
    np.testing.assert_array_less(-1e-10, pval_result.p_values)  # Allow small numerical errors
    np.testing.assert_array_less(pval_result.p_values, 1 + 1e-10)
    
    # Check BuildPumpProbeMasks output
    mask_result = results.results["build_masks"].output
    assert isinstance(mask_result, BuildPumpProbeMasksOutput)
    np.testing.assert_array_equal(mask_result.signal_mask.shape, (rows, cols))
    np.testing.assert_array_equal(mask_result.background_mask.shape, (rows, cols))
    assert mask_result.signal_mask.dtype == bool, "Signal mask should be boolean"
    assert mask_result.background_mask.dtype == bool, "Background mask should be boolean"
    # Masks shouldn't overlap
    np.testing.assert_array_equal(mask_result.signal_mask & mask_result.background_mask, 
                                np.zeros((rows, cols), dtype=bool))
