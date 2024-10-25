# btx/processing/tests/functional/test_pipeline_integration.py
import pytest
from pathlib import Path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from btx.processing.core.pipeline import Pipeline, PipelineBuilder
from btx.processing.core.adapters import registry
from btx.processing.tasks import (
    LoadData, MakeHistogram, MeasureEMD,
    CalculatePValues, BuildPumpProbeMasks, PumpProbeAnalysis
)
from typing import Dict, Any
from btx.processing.btx_types import LoadDataInput
from btx.processing.tests.functional.test_pump_probe import generate_synthetic_pump_probe_data  # Add this import

def generate_test_data_and_config():
   """Generate synthetic test data and configuration for the pipeline."""
   
   # Create configuration
   config = {
       'setup': {
           'run': 'synthetic_test',
           'exp': 'pipeline_validation',
           'background_roi_coords': [20, 40, 20, 40]
       },
       'load_data': {
           'roi': [0, 100, 0, 100],
           'time_bin': 2.0,
           'i0_threshold': 200
       },
       'make_histogram': {
           'bin_boundaries': np.arange(5, 30, 0.2),
           'hist_start_bin': 1
       },
       'calculate_emd': {
           'num_permutations': 1000
       },
       'calculate_pvalues': {
           'significance_threshold': 0.05
       },
       'generate_masks': {
           'threshold': 1.5,
           'bg_mask_mult': 2.0,
           'bg_mask_thickness': 5
       },
       'pump_probe_analysis': {
           'min_count': 10,
           'significance_level': 0.05
       }
   }

   # Generate synthetic data with exponential profile
   profile_params = {
       'amplitude': 0.3,
       't0': 0.0,
       'decay_time': 5.0
   }

   load_data_output, masks_output, _ = generate_synthetic_pump_probe_data(
       n_frames=2000,              
       rows=50,                    
       cols=50,
       delay_range=(-10, 20),      
       n_delay_bins=30,            
       noise_level=0.02,           
       signal_profile='exponential',
       profile_params=profile_params,
       min_frames_per_bin=20       
   )

   return config, load_data_output

def build_pipeline(config: Dict[str, Any], diagnostics_dir: Path) -> Pipeline:
   # Register all tasks
   registry.register("load_data", LoadData)
   registry.register("make_histogram", MakeHistogram)
   registry.register("measure_emd", MeasureEMD)
   registry.register("calculate_pvalues", CalculatePValues)
   registry.register("build_masks", BuildPumpProbeMasks)
   registry.register("pump_probe", PumpProbeAnalysis)
   
   # Build pipeline with correct task ordering and dependencies
   return (PipelineBuilder("Analysis Pipeline")
       .add("load_data", registry.create("load_data", config))
       .add("make_histogram", registry.create("make_histogram", config),
            ["load_data"])
       .add("measure_emd", registry.create("measure_emd", config),
            ["make_histogram"])
       .add("calculate_pvalues", registry.create("calculate_pvalues", config),
            ["measure_emd"])
       .add("build_masks", registry.create("build_masks", config),
            ["make_histogram", "calculate_pvalues"])
       .add("pump_probe", registry.create("pump_probe", config),
            ["load_data", "build_masks"])
       .set_diagnostics_dir(diagnostics_dir)
       .build())

def main():
   """Run end-to-end pipeline test with synthetic data."""
   # Setup output directory
   output_dir = Path("pipeline_results")
   output_dir.mkdir(exist_ok=True)
   diagnostics_dir = output_dir / "diagnostics"
   diagnostics_dir.mkdir(exist_ok=True)

   print("\nGenerating test data and configuration...")
   config, load_data_output = generate_test_data_and_config()
   
   print("\nBuilding pipeline...")
   pipeline = build_pipeline(config, diagnostics_dir)
   
   print("\nCreating pipeline input...")
   input_data = LoadDataInput(
       config=config,
       data=load_data_output.data,
       I0=load_data_output.I0,
       laser_delays=load_data_output.laser_delays,
       laser_on_mask=load_data_output.laser_on_mask,
       laser_off_mask=load_data_output.laser_off_mask
   )
   
   print("\nRunning pipeline...")
   results = pipeline.run(input_data)
   
   # Check pipeline success
   if not results.success:
       print(f"\nPipeline failed: {results.error}")
       for task_name, result in results.results.items():
           if not result.success:
               print(f"Task {task_name} failed: {result.error}")
       return
   
   print("\nPipeline completed successfully!")
   print(f"Results and diagnostics saved to: {output_dir}")
   
   # Print summary of results
   print("\nResults summary:")
   for task_name, result in results.results.items():
       print(f"\n{task_name}:")
       if isinstance(result.output, np.ndarray):
           print(f"  Shape: {result.output.shape}")
       elif hasattr(result.output, '__dict__'):
           print("  Attributes:", list(result.output.__dict__.keys()))
   
   return results

from pathlib import Path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from btx.processing.core.pipeline import Pipeline, PipelineBuilder
from btx.processing.core.adapters import registry
from btx.processing.tasks import (
    LoadData, MakeHistogram, MeasureEMD,
    CalculatePValues, BuildPumpProbeMasks, PumpProbeAnalysis
)
from btx.processing.btx_types import LoadDataInput

def test_pipeline_integration():
    """Test end-to-end pipeline with synthetic exponential decay data."""
    # Setup
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'pipeline_test'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    config, load_data_output = generate_test_data_and_config()
    
    # Build and run pipeline
    pipeline = build_pipeline(config, save_dir / "diagnostics")
    
    input_data = LoadDataInput(
        config=config,
        data=load_data_output.data,
        I0=load_data_output.I0,
        laser_delays=load_data_output.laser_delays,
        laser_on_mask=load_data_output.laser_on_mask,
        laser_off_mask=load_data_output.laser_off_mask
    )
    
    results = pipeline.run(input_data)
    
    # Validate pipeline execution
    assert results.success, f"Pipeline failed: {results.error}"
    expected_order = ["load_data", "make_histogram", "measure_emd", 
                     "calculate_pvalues", "build_masks", "pump_probe"]
    assert results.execution_order == expected_order, "Incorrect execution order"
    
    # Validate task outputs exist and have correct shapes/types
    load_output = results.results["load_data"].output
    assert load_output.data.shape == (2000, 50, 50), "Incorrect data shape"
    assert load_output.I0.shape == (2000,), "Incorrect I0 shape"
    
    hist_output = results.results["make_histogram"].output
    n_bins = len(config['make_histogram']['bin_boundaries']) - config['make_histogram']['hist_start_bin'] - 1
    assert hist_output.histograms.shape == (n_bins, 50, 50), "Incorrect histogram shape"
    
    emd_output = results.results["measure_emd"].output
    assert emd_output.emd_values.shape == (50, 50), "Incorrect EMD shape"
    
    pval_output = results.results["calculate_pvalues"].output
    assert pval_output.p_values.shape == (50, 50), "Incorrect p-values shape"
    
    masks_output = results.results["build_masks"].output
    assert masks_output.signal_mask.shape == (50, 50), "Incorrect mask shape"
    assert masks_output.signal_mask.dtype == bool, "Incorrect mask type"
    
    # Validate final pump-probe analysis
    pp_output = results.results["pump_probe"].output
    signal_diff = pp_output.signals_on - pp_output.signals_off
    
    # Fit exponential decay
    def exp_model(t, A, tau):
        return A * np.exp(-t/tau)
    
    pos_delays = pp_output.delays > 0
    popt, _ = optimize.curve_fit(
        exp_model, 
        pp_output.delays[pos_delays], 
        signal_diff[pos_delays],
        p0=[0.3, 5.0]  # Expected amplitude and decay time
    )
    fitted_tau = popt[1]
    expected_tau = 5.0
    
    # Validate decay time within 30%
    assert np.abs(fitted_tau - expected_tau) < expected_tau * 0.3, \
        f"Fitted decay time {fitted_tau:.1f} differs from expected {expected_tau:.1f}"
    
    # Generate summary plots
    plt.figure(figsize=(12, 8))
    
    # Input data visualization
    plt.subplot(221)
    plt.imshow(np.mean(input_data.data, axis=0))
    plt.title('Average Input Frame')
    plt.colorbar()
    
    # Mask visualization
    plt.subplot(222)
    combined_mask = np.zeros(masks_output.signal_mask.shape)
    combined_mask[masks_output.signal_mask] = 1
    combined_mask[masks_output.background_mask] = 2
    plt.imshow(combined_mask, cmap='viridis')
    plt.title('Signal (1) and Background (2) Masks')
    plt.colorbar()
    
    # Time trace visualization
    plt.subplot(223)
    plt.errorbar(pp_output.delays, signal_diff,
                yerr=np.sqrt(pp_output.std_devs_on**2 + pp_output.std_devs_off**2),
                fmt='o', label='Data')
    plt.plot(pp_output.delays[pos_delays],
            exp_model(pp_output.delays[pos_delays], *popt),
            'r-', label=f'Fit (τ = {fitted_tau:.1f} ps)')
    plt.xlabel('Delay (ps)')
    plt.ylabel('Signal Difference')
    plt.legend()
    plt.grid(True)
    
    # P-value visualization
    plt.subplot(224)
    plt.imshow(-np.log10(pval_output.p_values))
    plt.title('-log10(p-values)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'pipeline_summary.png')
    plt.close()
    
    # Verify diagnostic outputs exist
    for task_name in expected_order:
        diag_path = save_dir / "diagnostics" / task_name
        assert diag_path.exists(), f"Missing diagnostics for {task_name}"
        assert any(diag_path.iterdir()), f"No diagnostic plots for {task_name}"

if __name__ == '__main__':
    test_pipeline_integration()
