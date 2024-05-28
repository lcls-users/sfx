# xppPumpProbe Analysis Pipeline

*xppPumpProbe* is an analysis pipeline for analyzing pump-probe experiments at XPP. The pipeline processes and analyzes data from pump-probe experiments, including loading raw data using smd, generating histograms, calculating Earth Mover's Distance (EMD), computing p-values, generating masks, and performing pump-probe analysis. The data loader may have to be redefined per-experiment, but the other components are reusable as is.

## Pipeline Structure

The *xppPumpProbe* analysis pipeline consists of several tasks that are executed sequentially. Each task is implemented as a separate class in the `xppmask.py` file. The tasks are:

1. **LoadData**: Loads raw data for pump-probe analysis.
2. **MakeHistogram**: Generates normalized histograms from loaded data.
3. **MeasureEMD**: Calculates Earth Mover's Distance (EMD) between histograms and a reference histogram.
4. **CalculatePValues**: Calculates p-values from EMD values and a null distribution.
5. **BuildPumpProbeMasks**: Generates binary masks based on p-values and ROI coordinates.
6. **PumpProbeAnalysis**: Performs pump-probe analysis using the generated masks and data.

## Configuration

```yaml
setup: 
  root_dir: /path/to/data 
  exp: experiment_name 
  run: run_number 

load_data: 
  roi: [x_start, x_end, y_start, y_end] 
  energy_filter: [energy_center, energy_width] 
  i0_threshold: 200 
  ipm_pos_filter: [0.2, 0.5] 
  time_bin: 2 
  time_tool: [0., 0.005] 
  output_dir: /path/to/output 

make_histogram: 
  input_file: /path/to/loaded_data.npz 
  output_dir: /path/to/output 

measure_emd: 
  histograms_path: /path/to/histograms.npy 
  output_path: /path/to/output 
  roi_coords: [x_start, x_end, y_start, y_end] 
  num_permutations: 1000 

calculate_p_values: 
  emd_path: /path/to/emd_values.npy 
  null_dist_path: /path/to/null_dist.npy 
  output_path: /path/to/output 

generate_masks: 
  histograms_path: /path/to/histograms.npy 
  p_values_path: /path/to/p_values.npy 
  output_dir: /path/to/output 
  roi_coords: [x_start, x_end, y_start, y_end] 
  threshold: 0.1 
  bg_mask_mult: 2.0 
  bg_mask_thickness: 5 

pump_probe_analysis: 
  output_dir: /path/to/output 
```

## Task Descriptions

### LoadData

The `LoadData` task is responsible for loading raw data for pump-probe analysis. It uses the `get_imgs_thresh` function from the `xpploader.py` file to load the data based on the specified run number, experiment number, ROI, energy filter, I0 threshold, IPM position filter, time bin, and time tool parameters.

### MakeHistogram

The `MakeHistogram` task generates normalized histograms from the loaded data. It uses the `calculate_histograms` function from the `histogram_analysis.py` file to create the histograms. The generated histograms are saved as a numpy file, and a summary file with histogram information is created.

### MeasureEMD

The `MeasureEMD` task calculates the Earth Mover's Distance (EMD) between histograms and a reference histogram. It uses the `calculate_emd_values` function from the `pvalues.py` file to compute the EMD values. The EMD values are saved as a numpy file, and a null distribution of EMD values is generated and saved as a separate numpy file. A summary file with EMD calculation information is also created.

### CalculatePValues

The `CalculatePValues` task calculates p-values from EMD values and a null distribution. It uses the `calculate_p_values` function from the `analysis_tasks.py` file to compute the p-values. The p-values are saved as a numpy file, and a summary file with p-value calculation information is generated.

### BuildPumpProbeMasks

The `BuildPumpProbeMasks` task generates binary signal and background masks based on the p-values, histograms, and ROI coordinates. It uses the `generate_signal_mask` and `generate_background_mask` functions from the `analysis_tasks.py` file to create the masks. The binary signal and background masks are saved as separate numpy files, and a summary file with mask generation information is produced.

### PumpProbeAnalysis

The `PumpProbeAnalysis` task performs the pump-probe analysis using the generated masks and loaded data. It groups images into stacks, generates intensity vs. delay curves, calculates p-values, plots the results, and saves the output files. The task produces a pump-probe plot, pump-probe curves data, a summary file, and a report file.

## Running the Pipeline

To run the *xppPumpProbe* analysis pipeline, follow these steps:

1. Prepare the YAML configuration file with the necessary parameters for each task.
2. Create an instance of each task class in the desired order, passing the configuration dictionary to the constructor.
3. Call the appropriate methods of each task instance to load data, perform computations, summarize results, report results, and save outputs. The output files and plots will be generated in the specified output directories.

Example:

```python
import os
import numpy as np
import h5py
from xppmask import *
from xppmask import MakeHistogram
from xppmask import LoadData

def load_data(output_dir, setup_config):
    config = {
        'setup': setup_config,
        'load_data': {
            'roi': [5, 105, 50, 250],
            'output_dir': output_dir,
        },
    }

    load_data = LoadData(config)
    load_data.load_data()
    load_data.save_data()
    load_data.summarize()

    output_file = os.path.join(output_dir, f"{setup_config['exp']}_run{setup_config['run']}_data.npz")

    return output_file

def make_histogram(data_file, output_dir, setup_config):
    config = {
        'setup': {
            'exp': setup_config['exp'],
            'run': setup_config['run'],
        },
        'make_histogram': {
            'input_file': data_file,
            'output_dir': output_dir,
        },
    }

    make_histogram = MakeHistogram(config)
    make_histogram.load_data()
    make_histogram.generate_histograms()
    make_histogram.summarize()

    histogram_file = make_histogram.save_histograms()
    make_histogram.save_histogram_image()

    return histogram_file, make_histogram.histogram_summary

def make_histogram_from_npz(npz_file, output_dir, setup_config):
    config = {
        'setup': {
            'exp': setup_config['exp'],
            'run': setup_config['run'],
        },
        'make_histogram': {
            'input_file': npz_file,
            'output_dir': output_dir,
        },
    }

    make_histogram = MakeHistogram(config)
    
    with np.load(npz_file) as data:
        make_histogram.data = data['data']
    
    make_histogram.generate_histograms()
    make_histogram.summarize()

    histogram_file = make_histogram.save_histograms()
    make_histogram.save_histogram_image()

    return histogram_file, make_histogram.histogram_summary

def measure_emd(histogram_file, output_dir, setup_config):
    config = {
        'calculate_emd': {
            'histograms_path': histogram_file,
            'output_path': output_dir,
            'roi_coords': [50, 100, 0, 200],
            'num_permutations': 100,  # Reduced permutations for faster testing
        },
        'setup': {
            'exp': setup_config['exp'],
            'run': setup_config['run'],
        },
    }

    emd_task = MeasureEMD(config)
    emd_task.load_histograms()
    emd_task.calculate_emd()
    emd_task.summarize()
    report_path = os.path.join(emd_task.output_dir, 'report.txt')
    emd_task.report(report_path)
    emd_task.save_emd_values()
    emd_task.save_null_distribution()

    return os.path.join(emd_task.output_dir, "emd_values.npy"), os.path.join(emd_task.output_dir, "emd_null_dist.npy")

def calculate_p_values(emd_file, null_dist_file, output_dir, setup_config):
    config = {
        'calculate_p_values': {
            'emd_path': emd_file,
            'null_dist_path': null_dist_file,
            'output_path': output_dir,
        },
        'setup': {
            'exp': setup_config['exp'],
            'run': setup_config['run'],
        },
    }

    pval_task = CalculatePValues(config)
    pval_task.load_data()
    pval_task.calculate_p_values()
    pval_task.summarize()

    report_path = os.path.join(pval_task.output_dir, 'pvalues_report.txt')
    pval_task.report(report_path)
    pval_task.save_p_values()

    return os.path.join(pval_task.output_dir, "p_values.npy")

def build_pump_probe_masks(p_values_file, histograms_file, output_dir, setup_config):
    config = {
        'generate_masks': {
            'histograms_path': histograms_file,
            'p_values_path': p_values_file,
            'output_dir': output_dir,
            'roi_coords': [50, 100, 0, 200],
            'threshold': 0.1,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5,
        },
        'setup': {
            'exp': setup_config['exp'],
            'run': setup_config['run'],
        },
    }

    mask_task = BuildPumpProbeMasks(config)
    mask_task.load_histograms()
    mask_task.load_p_values()
    mask_task.generate_masks()
    mask_task.summarize()

    report_path = os.path.join(mask_task.output_dir, 'masks_report.txt')
    mask_task.report(report_path)
    signal_mask_file, bg_mask_file = mask_task.save_masks()

    return signal_mask_file, bg_mask_file

def run_pump_probe_analysis(config, data_file, signal_mask_file, bg_mask_file):
    data = np.load(data_file, allow_pickle=True)
    signal_mask = np.load(signal_mask_file)
    bg_mask = np.load(bg_mask_file)

    analysis = PumpProbeAnalysis(config)
    analysis.run(data['data'], data['binned_delays'], data['I0'], data['laser_on_mask'], data['laser_off_mask'], signal_mask, bg_mask)

    print("PumpProbeAnalysis completed successfully.")

if __name__ == '__main__':
    example_config = {
        'exp': 'xppx1003221',
        'run': 195,
        'setup': {
            'exp': 'xppx1003221',
            'run': 195,
            'root_dir': '/sdf/data/lcls/ds/xpp/xppx1003221', 
        },
        'pump_probe_analysis': {
            'output_dir': 'pump_probe_output',
            # Add other required configuration values for pump-probe analysis
        }
    }

    data_file = load_data('loaded_data', example_config)

    hfile_from_load_data, summary_from_load_data = make_histogram(data_file, 'histograms_from_load_data', example_config)
    print("MakeHistogram with LoadData output:")
    print(summary_from_load_data)

    emd_file, null_dist_file = measure_emd(hfile_from_load_data, 'emd_output', example_config)

    p_values_file = calculate_p_values(emd_file, null_dist_file, 'pvalues_output', example_config)

    signal_mask_file, bg_mask_file = build_pump_probe_masks(p_values_file, hfile_from_load_data, 'masks_output', example_config)

    run_pump_probe_analysis(example_config, data_file, signal_mask_file, bg_mask_file)
```

## Testing

The *xppPumpProbe* pipeline includes a test script located in the `tests/test_xppmask.py` file. The test script contains individual test functions for each task and an integration test function (`test_pump_probe_analysis`) that runs all tasks in sequence. To run the tests, execute the `test_xppmask.py` script.
