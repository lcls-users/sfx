# btx/processing/tests/functional/test_pipeline_integration.py
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
from btx.processing.btx_types import (
    LoadDataInput, LoadDataOutput,
    MakeHistogramInput, MakeHistogramOutput,
    MeasureEMDInput, MeasureEMDOutput,
    CalculatePValuesInput, CalculatePValuesOutput,
    BuildPumpProbeMasksInput, BuildPumpProbeMasksOutput,
    PumpProbeAnalysisInput, PumpProbeAnalysisOutput
)
from btx.processing.tests.functional.test_pump_probe import generate_synthetic_pump_probe_data
config = {
   'setup': {
       'run': 'synthetic_test',
       'exp': 'pipeline_validation',
       'background_roi_coords': [0, 15, 0, 40]
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
       'threshold': 0.05, # assert between 0 and 1
       'bg_mask_mult': 2.0,
       'bg_mask_thickness': 5
   },
   'pump_probe_analysis': {
       'min_count': 10,
       'significance_level': 0.05
   }
}

# Setup output directory
output_dir = Path("pipeline_results")
output_dir.mkdir(exist_ok=True)
diagnostics_dir = output_dir / "diagnostics"
diagnostics_dir.mkdir(exist_ok=True)

print("\nGenerating test data and configuration...")
load_data_output, masks_output, signal_values = generate_synthetic_pump_probe_data(signal_profile = 'exponential',
                                                         profile_params={'amplitude': 5, 'decay_time': 5.0}
)

print("\nCreating histogram...")
make_histogram = MakeHistogram(config)
histogram_input = MakeHistogramInput(
    config=config,
    load_data_output=load_data_output
)
histogram_output = make_histogram.run(histogram_input)
make_histogram.plot_diagnostics(histogram_output, diagnostics_dir / "make_histogram")

print("\nMeasuring EMD...")
measure_emd = MeasureEMD(config)
emd_input = MeasureEMDInput(
    config=config,
    histogram_output=histogram_output
)
emd_output = measure_emd.run(emd_input)
measure_emd.plot_diagnostics(emd_output, diagnostics_dir / "measure_emd")

print("\nCalculating p-values...")
calculate_pvalues = CalculatePValues(config)
pvalues_input = CalculatePValuesInput(
    config=config,
    emd_output=emd_output
)
pvalues_output = calculate_pvalues.run(pvalues_input)
calculate_pvalues.plot_diagnostics(pvalues_output, diagnostics_dir / "calculate_pvalues")

print("\nBuilding masks...")
build_masks = BuildPumpProbeMasks(config)
masks_input = BuildPumpProbeMasksInput(
    config=config,
    histogram_output=histogram_output,
    p_values_output=pvalues_output
)
masks_output = build_masks.run(masks_input)
build_masks.plot_diagnostics(masks_output, diagnostics_dir / "build_masks")

print("\nRunning pump-probe analysis...")
pump_probe = PumpProbeAnalysis(config)
pump_probe_input = PumpProbeAnalysisInput(
    config=config,
    load_data_output=load_data_output,
    masks_output=masks_output
)
pump_probe_output = pump_probe.run(pump_probe_input)
pump_probe.plot_diagnostics(pump_probe_output, diagnostics_dir / "pump_probe")

print("\nProcessing complete! Results saved in:", output_dir)

# Print some key results
print("\nKey Results:")
print(f"Number of significant pixels: {np.sum(pvalues_output.p_values < config['calculate_pvalues']['significance_threshold'])}")
print(f"Signal mask size: {np.sum(masks_output.signal_mask)} pixels")
print(f"Background mask size: {np.sum(masks_output.background_mask)} pixels")
print(f"Number of delay points: {len(pump_probe_output.delays)}")
print(f"Mean I0 (laser on): {pump_probe_output.mean_I0_on:.1f}")
print(f"Mean I0 (laser off): {pump_probe_output.mean_I0_off:.1f}")
