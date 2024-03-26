from build_pump_probe_masks import BuildPumpProbeMasks
from calculate_p_values import CalculatePValues 
from make_histogram import MakeHistogram
from measure_emd import MeasureEMD
import numpy as np
import os
import tempfile

def test_make_histogram(data_stack, det_type='GENERIC'):
    """
    Functional test for the MakeHistogram class.

    Args:
        data_stack (ndarray): 3D array of pixel values (frames, rows, cols)
        det_type (str, optional): Detector type for preprocessing. Defaults to 'GENERIC'.

    Returns:
        dict: Summary of the generated histograms
    """
    # Create a temporary configuration dictionary
    config = {
        'setup': {
            'root_dir': '.',
            'exp': 'test_exp',
            'run': 1,
            'det_type': det_type,
        },
        'make_histogram': {
            'input_file': 'test_data.h5',
            'dataset': '/entry/data/data',
            'output_dir': '.',
        },
    }

    # Save the data stack to a temporary HDF5 file
    with h5py.File('test_data.h5', 'w') as hdf5_file:
        hdf5_file.create_dataset('/entry/data/data', data=data_stack)

    # Create and run the MakeHistogram instance
    make_histogram = MakeHistogram(config)
    make_histogram.load_data()
    make_histogram.generate_histograms()
    make_histogram.summarize()

    # Return the histogram summary for inspection
    return make_histogram.histogram_summary

# test_xppmask.py


def test_measure_emd():
    with tempfile.TemporaryDirectory() as temp_dir:
        histograms = np.random.rand(10, 100, 100) 
        hist_path = os.path.join(temp_dir, 'histograms.npy')
        np.save(hist_path, histograms)

        config = {
            'emd': {
                'histograms_path': hist_path,
                'output_path': os.path.join(temp_dir, 'emd_values.npy'),
                'roi_coords': [40, 60, 40, 60]
            },
            'setup': {
                'exp': 'test_exp',
                'run': 'test_run'  
            }
        }

        emd_task = MeasureEMD(config)
        emd_task.load_histograms()
        emd_task.calculate_emd()
        emd_task.summarize()
        report_path = os.path.join(temp_dir, 'report.txt')
        emd_task.report(report_path)
        emd_task.save_emd_values()

        assert os.path.exists(report_path)
        assert os.path.exists(config['emd']['output_path'])

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
                'output_path': os.path.join(temp_dir, 'p_values.npy')
            }, 
            'setup': {
                'exp': 'test_exp',
                'run': 'test_run'
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
                'bg_mask_thickness': 5
            },
            'setup': { 
                'exp': 'test_exp',
                'run': 'test_run'
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
    # Example usage
    data_stack = np.random.rand(100, 16, 16)

    summary = test_make_histogram(data_stack)
    print(summary)

#from test_make_histogram import test_make_histogram
#
#data_stack = ...  # Your 3D numpy array
#
#summary = test_make_histogram(data_stack)
#print(summary)
