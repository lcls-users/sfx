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


def test_measure_emd(histogram_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'emd': {
                'histograms_path': histogram_file,
                'output_dir': temp_dir,
                'roi_coords': [40, 60, 40, 60],
                'num_permutations': 100,  # Reduced permutations for faster testing
            },
            'setup': {
                'exp': 'test_exp',
                'run': 'test_run',
            }
        }

        emd_task = MeasureEMD(config)
        emd_task.load_histograms()
        emd_task.calculate_emd()
        emd_task.generate_null_distribution()  # Add this line
        emd_task.summarize()
        
        report_path = os.path.join(temp_dir, 'report.txt')
        emd_task.report(report_path)
        emd_task.save_emd_values()
        emd_task.save_null_distribution()

        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(temp_dir, "emd_values_test_exp_test_run.npy"))
        assert os.path.exists(os.path.join(temp_dir, "emd_null_dist_test_exp_test_run.npy"))

def test_calculate_p_values():
    with tempfile.TemporaryDirectory() as temp_dir:
        emd_values = np.random.rand(100, 100)
        emd_path = os.path.join(temp_dir, 'emd_values.npy')
        np.save(emd_path, emd_values)

        null_dist = np.random.rand(1000)
        null_dist_path = os.path.join(temp_dir, 'null_dist.npy')
        np.save(null_dist_path, null_dist)

        config = {
            'pvalues': {
                'emd_path': emd_path,
                'null_dist_path': null_dist_path,
                'output_path': os.path.join(temp_dir, 'p_values.npy'),
            },
            'setup': {
                'exp': 'test_exp',
                'run': 'test_run',
            }
        }

        pval_task = CalculatePValues(config)
        pval_task.load_data()
        pval_task.calculate_p_values()
        pval_task.summarize()
        report_path = os.path.join(temp_dir, 'report.txt')
        pval_task.report(report_path)
        pval_task.save_p_values()

        assert os.path.exists(report_path)
        assert os.path.exists(config['pvalues']['output_path'])

def test_build_pump_probe_masks():
    with tempfile.TemporaryDirectory() as temp_dir:
        p_values = np.random.rand(100, 100)
        p_values_path = os.path.join(temp_dir, 'p_values.npy')
        np.save(p_values_path, p_values)

        config = {
            'masks': {
                'p_values_path': p_values_path,
                'output_dir': temp_dir,
                'roi_coords': [40, 60, 40, 60],
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
        report_path = os.path.join(temp_dir, 'report.txt')
        mask_task.report(report_path)
        mask_task.save_masks()

        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(temp_dir, 'signal_mask.npy'))
        assert os.path.exists(os.path.join(temp_dir, 'bg_mask.npy'))

if __name__ == '__main__':
    data = np.load('/home/ollie/data.npz')['arr_0']
    hfile, summary = test_make_histogram(data, '.')
    print(summary)
    test_measure_emd(hfile)

