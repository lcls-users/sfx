# XPP Analysis Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

The purpose of this package is to do signal detection, statistical testing, and end-to-end analysis of time series datasets collected from X ray pump-probe experiments at the LCLS.

## Key Features

- **Data Processing Pipeline**
  - HDF5 experimental file integration
  - Automated mask generation
  - Histogram-based signal analysis
  - Earth Mover's Distance (EMD) calculations
  - Statistical significance testing
  - Pump-probe time series analysis

- **Performance Features**
  - Numba-accelerated computations
  - Memoization for repeated calculations
  - Efficient memory management

- **Diagnostics**
  - Visual analysis at each pipeline stage
  - Statistical validation reporting
  - Performance profiling
  - Automated quality checks

## Quick Start

### Basic Configuration

```python
import numpy as np
from pathlib import Path
from btx.processing.tasks import (
    LoadData, MakeHistogram, MeasureEMD,
    CalculatePValues, BuildPumpProbeMasks, PumpProbeAnalysis
)

# Configure the analysis pipeline
config = {
    'setup': {
        'run': 190,
        'exp': 'xppl1030522',
        'background_roi_coords': [0, 15, 0, 40]  # Background ROI for analysis
    },
    'load_data': {
        'roi': (170, 250, 135, 215),  # Region of interest for processing
        'energy_filter': [9.0, 5.0],   # Energy filtering parameters
        'i0_threshold': 0,             # Minimum I0 value
        'time_bin': 2.0,               # Time binning in ps
        'time_tool': [0.0, 0.015]      # TimeTool parameters
    },
    'make_histogram': {
        'bin_boundaries': np.arange(5, 30, 0.2),
        'hist_start_bin': 1
    },
    'calculate_emd': {
        'num_permutations': 1000       # Number of permutations for null distribution
    },
    'calculate_pvalues': {
        'significance_threshold': 0.05  # P-value threshold for significance
    },
    'generate_masks': {
        'threshold': 0.05,             # P-value threshold for mask generation
        'bg_mask_mult': 2.0,           # Background mask size multiplier
        'bg_mask_thickness': 5         # Background mask thickness in pixels
    },
    'pump_probe_analysis': {
        'min_count': 2,                # Minimum frames per delay bin
        'significance_level': 0.05,     # P-value threshold for significance
        'Emin': 7.0,                   # Minimum energy threshold (keV)
        'Emax': float('inf')           # Maximum energy threshold (keV)
    }
}
```


## Pipeline Components

The pump-probe diffraction analysis pipeline consists of the following stages, connected as shown below:

```
            LoadData
            /      \
   MakeHistogram   \
       /    \       \
MeasureEMD  |       \
     |      |        \
CalculatePValues     |
     \      /        |
BuildPumpProbeMasks  |
           \         |
          PumpProbeAnalysis
```
Why this structure? First, it allows caching and visual inspection of intermediate results. If we want to run an optimization loop with respect to input (configuration) parameters of the signal-finding task, for example, it is necessary to memoize CalculatePValues or its expensive upstream tasks. (The actual current behavior is that MakeHistogram, the most expensive step, is memoized by default.) 

We *could* do these things while black boxing the entire pipeline, but exposing the intermediate steps as a DAG - with explicit inputs, outputs and visual monitoring for each component - gives the user more clarity and control over what is going on. 

Second, our goal is to provide an extensible library, not a monolithic analysis script. Because the tasks are explicitly composable and have well-defined, stateless behavior, it is much easier to add new capability. For example, extending the pump-probe analysis to multiple diffraction peaks would simply require a customized replacement of BuildPumpProbeMasks that returns multiple signal masks instead of a single one. The user would then just run PumpProbeAnalysis on each of those outputs. 

1. **Data Loading** (`LoadData`)
   - Input: Raw experimental data (frames, I0 values, laser delays, masks)
   - Output: Preprocessed data array, binned delays, validated masks
   - Key operations:
     - Raw data ingestion
     - Preprocessing
     - Initial validation

2. **Histogram Generation** (`MakeHistogram`)
   - Input: LoadData output (frames and metadata)
   - Output: Per-pixel histograms, bin edges, bin centers
   - Key operations:
     - Numba-optimized computation
     - Configurable binning
     - Memoization

3. **EMD Calculation** (`MeasureEMD`)
   - Input: MakeHistogram output (histograms and bin information)
   - Output: EMD values per pixel, null distribution
   - Key operations:
     - Background ROI validation
     - Wasserstein distance computation
     - Null distribution generation

4. **Statistical Analysis** (`CalculatePValues`)
   - Input: MeasureEMD output (EMD values and null distribution)
   - Output: P-values and log-transformed p-values
   - Key operations:
     - P-value calculation
     - Multiple testing correction
     - Significance thresholding

5. **Signal-finding / Mask Generation** (`BuildPumpProbeMasks`)
   - Input: MakeHistogram output and CalculatePValues output
   - Output: Signal mask, background mask, intermediate mask stages
   - Key operations:
     - ROI-connected clustering
     - Buffer zone generation
     - Quality validation

6. **Time Series Analysis** (`PumpProbeAnalysis`)
   - Input: LoadData output and BuildPumpProbeMasks output
   - Output: Time-dependent signals, uncertainties, frame statistics
   - Key operations:
     - Delay-based grouping
     - Signal calculation
     - Error propagation

### Code example

```python
# Setup output directory for results and diagnostics
output_dir = Path("pipeline_results")
output_dir.mkdir(exist_ok=True)
diagnostics_dir = output_dir / "diagnostics"
diagnostics_dir.mkdir(exist_ok=True)

# Load and process data
def load_pump_probe_data(npz_path, roi, config):
    """Load and preprocess pump-probe data."""
    with np.load(npz_path) as data:
        frames = data['frames'][:, roi[2]:roi[3], roi[0]:roi[1]]
        
        loader = LoadData(config)
        return loader.process(
            config=config,
            data=frames,
            I0=data['I0'],
            laser_delays=data['delays'],
            laser_on_mask=data['laser_on_mask'],
            laser_off_mask=data['laser_off_mask']
        )

# Process through pipeline
load_data_output = load_pump_probe_data(
    'path/to/your/data.npz',
    roi=config['load_data']['roi'],
    config=config
)

# Generate histograms
histogram = MakeHistogram(config)
histogram_output = histogram.process(
    config=config,
    load_data_output=load_data_output
)

# Calculate Earth Mover's Distance
emd = MeasureEMD(config)
emd_output = emd.process(
    config=config,
    histogram_output=histogram_output
)

# Calculate statistical significance
pvals = CalculatePValues(config)
pvals_output = pvals.process(
    config=config,
    emd_output=emd_output
)

# Generate analysis masks
masks = BuildPumpProbeMasks(config)
masks_output = masks.process(
    config=config,
    histogram_output=histogram_output,
    p_values_output=pvals_output
)

# Perform pump-probe analysis
analysis = PumpProbeAnalysis(config)
results = analysis.process(
    config=config,
    load_data_output=load_data_output,
    masks_output=masks_output
)
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Numba
- H5py
- PyTables

## Future Work

### File I/O Standardization

We might implement a standardized file organization system for task inputs and outputs. Each pipeline stage will follow consistent conventions for data storage and retrieval:

```
output_dir/
├── load_data/
│   └── {exp}_{run}/
│       └── {exp}_run{run}_data.npz
├── make_histogram/
│   └── {exp}_{run}/
│       └── histograms.npy
├── measure_emd/
│   └── {exp}_{run}/
│       ├── emd_values.npy
│       └── emd_null_dist.npy
├── calculate_p_values/
│   └── {exp}_{run}/
│       └── p_values.npy
├── build_pump_probe_masks/
│   └── {exp}_{run}/
│       ├── signal_mask.npy
│       └── bg_mask.npy
└── pump_probe_analysis/
    └── {exp}_{run}/
        └── pump_probe_curves.npz
```

#### Planned Improvements

1. **File Manager Interface**
   - Implement a centralized FileManager class to handle all I/O operations
   - Standardize path generation and file access across tasks
   - Add validation for file existence and data integrity

2. **Consistent Metadata**
   - Include metadata in saved files (timestamps, configuration parameters, etc.)
   - Implement versioning for data formats
   - Add checksums for data validation
