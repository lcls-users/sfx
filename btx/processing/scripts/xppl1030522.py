from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any
from btx.processing.btx_types import LoadDataOutput, LoadDataInput
from btx.processing.tasks import (
    LoadData, MakeHistogram, MeasureEMD,
    CalculatePValues, BuildPumpProbeMasks, PumpProbeAnalysis
)
from btx.processing.btx_types import (
    MakeHistogramInput, MeasureEMDInput,
    CalculatePValuesInput, BuildPumpProbeMasksInput,
    PumpProbeAnalysisInput
)

def load_pump_probe_data(
    npz_path: str,
    roi: Tuple[int, int, int, int],
    config: Dict[str, Any]
) -> Tuple[LoadDataOutput, np.ndarray]:
    """
    Load pump-probe data from npz file and process through LoadData.
    
    Args:
        npz_path: Path to npz file containing extracted data
        roi: Region of interest (x1, x2, y1, y2) to crop frames
        config: Configuration dictionary
    
    Returns:
        Tuple of (LoadDataOutput, frames)
    """
    print(f"Loading data from {npz_path}")
    
    # Load npz data
    with np.load(npz_path) as data:
        frames = data['frames']
        delays = data['delays']
        I0 = data['I0']
        laser_on_mask = data['laser_on_mask']
        laser_off_mask = data['laser_off_mask']
    
    print('frames shape', frames.shape)
    # Apply ROI cropping
    x1, x2, y1, y2 = roi
    print('roi', roi)
    frames = frames[:, y1:y2, x1:x2]
    
    # Use LoadData to properly bin the delays
    loader = LoadData(config)
    load_data_output = loader.process(
        config=config,
        data=frames,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=laser_on_mask,
        laser_off_mask=laser_off_mask
    )
    
    return load_data_output, frames

# Configuration
config = {
    'setup': {
        'run': 190,
        'exp': 'xppl1030522',
        'background_roi_coords': [0, 15, 0, 40]
    },
    'load_data': {
        'roi': (170, 250, 135, 215),  # Moved from function call to config
        'energy_filter': [9.0, 5.0],
        'i0_threshold': 0,
        'time_bin': 2.0,
        'time_tool': [0.0, 0.015]
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
        'threshold': 0.05,
        'bg_mask_mult': 2.0,
        'bg_mask_thickness': 5
    },
    'pump_probe_analysis': {
        'min_count': 2,
        'significance_level': 0.05
    }
}

# Setup output directory
output_dir = Path("pipeline_results")
output_dir.mkdir(exist_ok=True)
diagnostics_dir = output_dir / "diagnostics"
diagnostics_dir.mkdir(exist_ok=True)

# Load and process data
npz_path = '/sdf/data/lcls/ds/xpp/xppx1003221/results/ohoidn/Shift 4/processed_data/run0190_extracted.npz'
load_data_output, frames = load_pump_probe_data(
    npz_path,
    roi=config['load_data']['roi'],
    config=config
)

print("\nCreating histogram...")
make_histogram = MakeHistogram(config)
histogram_output = make_histogram.process(
    config=config,
    load_data_output=load_data_output
)
make_histogram.plot_diagnostics(histogram_output, diagnostics_dir / "make_histogram")

print("\nMeasuring EMD...")
measure_emd = MeasureEMD(config)
emd_output = measure_emd.process(
    config=config,
    histogram_output=histogram_output
)
measure_emd.plot_diagnostics(emd_output, diagnostics_dir / "measure_emd")

print("\nCalculating p-values...")
calculate_pvalues = CalculatePValues(config)
pvalues_output = calculate_pvalues.process(
    config=config,
    emd_output=emd_output
)
calculate_pvalues.plot_diagnostics(pvalues_output, diagnostics_dir / "calculate_pvalues")

print("\nBuilding masks...")
build_masks = BuildPumpProbeMasks(config)
masks_output = build_masks.process(
    config=config,
    histogram_output=histogram_output,
    p_values_output=pvalues_output
)
build_masks.plot_diagnostics(masks_output, diagnostics_dir / "build_masks")

print("\nRunning pump-probe analysis...")
pump_probe = PumpProbeAnalysis(config)
pump_probe_output = pump_probe.process(
    config=config,
    load_data_output=load_data_output,
    masks_output=masks_output
)
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
