# XPP Analysis Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This is a Python framework for analyzing X-ray Pump-Probe experimental data at light source facilities. It provides a pipeline for processing, analyzing, and validating experimental results using histogram-based signal analysis and Earth Mover's Distance (EMD) calculations.

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

## Installation

```bash
# Clone the repository
git clone git@github.com:hoidn/btx.git

# Navigate to the project directory
cd btx

# Install the package in development mode
pip install -e .
```

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
        'time_bin': 2.0,               # Time binning in ps
        'time_tool': [0.0, 0.015]      # TimeTool parameters
    },
    'make_histogram': {
        'bin_boundaries': np.arange(5, 30, 0.2),
        'hist_start_bin': 1
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

# Setup output directory for results and diagnostics
output_dir = Path("pipeline_results")
output_dir.mkdir(exist_ok=True)
diagnostics_dir = output_dir / "diagnostics"
diagnostics_dir.mkdir(exist_ok=True)
```

### Loading and Processing Data

```python
# Load data from NPZ file
from btx.processing.btx_types import LoadDataInput

def load_pump_probe_data(npz_path, roi, config):
    """Load and preprocess pump-probe data."""
    with np.load(npz_path) as data:
        frames = data['frames'][:, roi[2]:roi[3], roi[0]:roi[1]]
        
        return LoadData(config).run(LoadDataInput(
            config=config,
            data=frames,
            I0=data['I0'],
            laser_delays=data['delays'],
            laser_on_mask=data['laser_on_mask'],
            laser_off_mask=data['laser_off_mask']
        ))

# Process through pipeline
load_data_output, _ = load_pump_probe_data(
    'path/to/your/data.npz',
    roi=config['load_data']['roi'],
    config=config
)

# Generate histograms
histogram = MakeHistogram(config)
histogram_output = histogram.run(MakeHistogramInput(
    config=config,
    load_data_output=load_data_output
))

# Calculate Earth Mover's Distance
emd = MeasureEMD(config)
emd_output = emd.run(MeasureEMDInput(
    config=config,
    histogram_output=histogram_output
))

# Calculate statistical significance
pvals = CalculatePValues(config)
pvals_output = pvals.run(CalculatePValuesInput(
    config=config,
    emd_output=emd_output
))

# Generate analysis masks
masks = BuildPumpProbeMasks(config)
masks_output = masks.run(BuildPumpProbeMasksInput(
    config=config,
    histogram_output=histogram_output,
    p_values_output=pvals_output
))

# Perform pump-probe analysis
analysis = PumpProbeAnalysis(config)
results = analysis.run(PumpProbeAnalysisInput(
    config=config,
    load_data_output=load_data_output,
    masks_output=masks_output
))
```

## Pipeline Components

The BTX pipeline consists of the following stages:

1. **Data Loading** (`LoadData`)
   - Raw data ingestion
   - Preprocessing
   - Initial validation

2. **Histogram Generation** (`MakeHistogram`)
   - Numba-optimized computation
   - Configurable binning
   - Memoization

3. **EMD Calculation** (`MeasureEMD`)
   - Background ROI validation
   - Wasserstein distance computation
   - Null distribution generation

4. **Statistical Analysis** (`CalculatePValues`)
   - P-value calculation
   - Multiple testing correction
   - Significance thresholding

5. **Mask Generation** (`BuildPumpProbeMasks`)
   - ROI-connected clustering
   - Buffer zone generation
   - Quality validation

6. **Time Series Analysis** (`PumpProbeAnalysis`)
   - Delay-based grouping
   - Signal calculation
   - Error propagation

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Numba
- H5py
- PyTables
