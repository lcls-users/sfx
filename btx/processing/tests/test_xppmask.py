import os
import numpy as np
import h5py
from xppmask import *
from xppmask import MakeHistogram
from xppmask import LoadData

def test_load_data(output_dir, setup_config):
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
    assert os.path.exists(output_file), f"Output file {output_file} does not exist"

    return output_file

def test_make_histogram(data_stack, output_dir, setup_config):
    # Create a temporary configuration dictionary
    config = {
        'setup': {
            'exp': setup_config['exp'],
            'run': setup_config['run'],
        },
        'make_histogram': {
            'input_file': os.path.join(output_dir, 'test_data.h5'),
            'dataset': '/entry/data/data',
            'output_dir': output_dir,
        },
    }

    # Save the data stack to a temporary HDF5 file
    with h5py.File(config['make_histogram']['input_file'], 'w') as hdf5_file:
        hdf5_file.create_dataset('/entry/data/data', data=data_stack)

    # Create and run the MakeHistogram instance
    make_histogram = MakeHistogram(config)
    make_histogram.load_data()
    make_histogram.generate_histograms()
    make_histogram.summarize()

    # Save the histograms using the save_histograms() method
    make_histogram.save_histograms()

    # Get the path to the saved histogram file
    histogram_file = os.path.join(make_histogram.output_dir, "histograms.npy")

    # Check if the histogram file exists
    assert os.path.exists(histogram_file), f"Histogram file {histogram_file} does not exist"

    # Return the path to the histogram file and the summary
    return histogram_file, make_histogram.histogram_summary

def test_measure_emd(histogram_file, output_dir, setup_config):
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
    try:
        emd_task.calculate_emd()
    except Exception as e:
        print(f"Error in calculate_emd: {e}")
        raise

    emd_task.summarize()
    report_path = os.path.join(emd_task.output_dir, 'report.txt')
    emd_task.report(report_path)

    try:
        emd_task.save_emd_values()
    except Exception as e:
        print(f"Error in save_emd_values: {e}")
        raise

    try:
        emd_task.save_null_distribution()
    except Exception as e:
        print(f"Error in save_null_distribution: {e}")
        raise

    assert os.path.exists(report_path)
    assert os.path.exists(os.path.join(emd_task.output_dir, "emd_values.npy"))
    assert os.path.exists(os.path.join(emd_task.output_dir, "emd_null_dist.npy"))

    return os.path.join(emd_task.output_dir, "emd_values.npy"), os.path.join(emd_task.output_dir, "emd_null_dist.npy")

def test_calculate_p_values(emd_file, null_dist_file, output_dir, setup_config):
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

    assert os.path.exists(report_path)
    assert os.path.exists(os.path.join(pval_task.output_dir, "p_values.npy"))

    return os.path.join(pval_task.output_dir, "p_values.npy")

def test_build_pump_probe_masks(p_values_file, histograms_file, output_dir, setup_config):
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
    mask_task.save_masks()

    assert os.path.exists(report_path)
    assert os.path.exists(os.path.join(mask_task.output_dir, "signal_mask.npy"))
    assert os.path.exists(os.path.join(mask_task.output_dir, "bg_mask.npy"))

if __name__ == '__main__':
    # TODO test data loader task, but this will only work on s3df
    # if running in a different environment, download sample data instead
    data = np.load('data.npz')['arr_0']
    output_dir = '.'
    setup_config = {
        'exp': 'test_exp',
        'run': 1,
    }

    hfile, summary = test_make_histogram(data, output_dir, setup_config)
    print(summary)

    emd_file, null_dist_file = test_measure_emd(hfile, output_dir, setup_config)
    p_values_file = test_calculate_p_values(emd_file, null_dist_file, output_dir, setup_config)
    test_build_pump_probe_masks(p_values_file, hfile, output_dir, setup_config)
