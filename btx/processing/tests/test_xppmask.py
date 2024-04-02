import tempfile
import os
import numpy as np
import h5py
from xppmask import *

def test_make_histogram(data_stack, output_dir, det_type='GENERIC'):
    # Create a temporary configuration dictionary
    config = {
        'setup': {
            'exp': 'test_exp',
            'run': 1,
            'det_type': det_type,
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
    histogram_file = os.path.join(output_dir, f"histograms_{make_histogram.exp}_{make_histogram.run}.npy")

    # Check if the histogram file exists
    assert os.path.exists(histogram_file), f"Histogram file {histogram_file} does not exist"

    # Return the path to the histogram file and the summary
    return histogram_file, make_histogram.histogram_summary

def test_measure_emd(histogram_file, output_dir):
    config = {
        'emd': {
            'histograms_path': histogram_file,
            'output_dir': output_dir,
            'roi_coords': [50, 100, 0, 200],
            'num_permutations': 100,  # Reduced permutations for faster testing
        },
        'setup': {
            'exp': 'test_exp',
            'run': 'test_run',
        }
    }

    emd_task = MeasureEMD(config)
    emd_task.load_histograms()
    
    try:
        emd_task.calculate_emd()
    except Exception as e:
        print(f"Error in calculate_emd: {e}")
        raise

    try:
        emd_task.generate_null_distribution()
    except Exception as e:
        print(f"Error in generate_null_distribution: {e}")
        raise

    emd_task.summarize()
    
    report_path = os.path.join(output_dir, 'report.txt')
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
    assert os.path.exists(os.path.join(output_dir, "emd_values.npy"))
    assert os.path.exists(os.path.join(output_dir, "emd_null_dist.npy"))

    return os.path.join(output_dir, "emd_values.npy"), os.path.join(output_dir, "emd_null_dist.npy")

def test_calculate_p_values(emd_file, null_dist_file, output_dir):
    config = {
        'pvalues': {
            'emd_path': emd_file,
            'null_dist_path': null_dist_file,
            'output_path': os.path.join(output_dir, 'p_values.npy'),
        },
        'setup': {
            'exp': 'test_exp',
            'run': 'test_run',
        }
    }

    null_distribution = np.load(null_dist_file)  # Load the null_distribution from file

    pval_task = CalculatePValues(config, null_distribution)  # Pass null_distribution to CalculatePValues
    pval_task.load_data()
    pval_task.calculate_p_values()
    pval_task.summarize()
    report_path = os.path.join(output_dir, 'pvalues_report.txt')
    pval_task.report(report_path)
    pval_task.save_p_values()

    assert os.path.exists(report_path)
    assert os.path.exists(config['pvalues']['output_path'])

    return config['pvalues']['output_path']

def test_build_pump_probe_masks(p_values_file, output_dir):
    config = {
        'masks': {
            'p_values_path': p_values_file,
            'output_dir': output_dir,
            'roi_coords': [50, 100,  0, 200],
            'threshold': 0.1,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5,
        },
        'setup': {
            'exp': 'test_exp',
            'run': 'test_run',
        }
    }

    mask_task = BuildPumpProbeMasks(config)
    mask_task.load_p_values()
    mask_task.generate_masks()
    mask_task.summarize()
    report_path = os.path.join(output_dir, 'masks_report.txt')
    mask_task.report(report_path)
    mask_task.save_masks()

    assert os.path.exists(report_path)
    assert os.path.exists(os.path.join(output_dir, 'signal_mask_test_exp_test_run.npy'))
    assert os.path.exists(os.path.join(output_dir, 'bg_mask_test_exp_test_run.npy'))

if __name__ == '__main__':
    data = np.load('/home/ollie/data.npz')['arr_0']
    output_dir = '.'
    hfile, summary = test_make_histogram(data, output_dir)
    print(summary)
    emd_file, null_dist_file = test_measure_emd(hfile, output_dir)
    p_values_file = test_calculate_p_values(emd_file, null_dist_file, output_dir)
    test_build_pump_probe_masks(p_values_file, output_dir)

