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

import os
import numpy as np
import json
from xppmask import PumpProbeAnalysis, LoadData, BuildPumpProbeMasks, MakeHistogram, MeasureEMD, CalculatePValues

def test_pump_probe_analysis(config, data_file, signal_mask_file, bg_mask_file):
    # Load the output data from LoadData and BuildPumpProbeMasks
    data = np.load(data_file)
    signal_mask = np.load(signal_mask_file)
    bg_mask = np.load(bg_mask_file)

    # Initialize and run PumpProbeAnalysis
    analysis = PumpProbeAnalysis(config)
    analysis.run(data, signal_mask, bg_mask)

    # Check that output files were generated
    assert os.path.exists(analysis.output_dir), f"Output directory {analysis.output_dir} was not created"

    expected_files = ['pump_probe_plot.png', 'pump_probe_curves.npz', 'pump_probe_summary.txt', 'pump_probe_report.txt'] 
    for file in expected_files:
        file_path = os.path.join(analysis.output_dir, file)
        assert os.path.isfile(file_path), f"Expected output file {file_path} was not generated"

    # Load and check pump_probe_curves.npz
    curves = np.load(os.path.join(analysis.output_dir, 'pump_probe_curves.npz'))
    expected_keys = ['delays', 'signals_on', 'std_devs_on', 'signals_off', 'std_devs_off', 'p_values']
    for key in expected_keys:
        assert key in curves, f"Expected key {key} not found in pump_probe_curves.npz"
        assert curves[key].shape == (analysis.delays.size,), f"Unexpected shape for {key} in pump_probe_curves.npz"

    # Check pump_probe_summary.txt
    with open(os.path.join(analysis.output_dir, 'pump_probe_summary.txt'), 'r') as f:
        summary = f.read()
    assert f"Pump-probe analysis for {config['setup']['exp']} run {config['setup']['run']}:" in summary
    assert f"Number of delays: {analysis.delays.size}" in summary
    assert f"Results saved to: {analysis.output_dir}" in summary

    # Check pump_probe_report.txt
    with open(os.path.join(analysis.output_dir, 'pump_probe_report.txt'), 'r') as f:  
        report = f.read()
    assert f"Signal mask loaded from: {config['pump_probe_analysis']['signal_mask_path']}" in report
    assert f"Background mask loaded from: {config['pump_probe_analysis']['bg_mask_path']}" in report
    assert "Laser on signals:" in report
    assert "Laser off std devs:" in report
    assert "p-values:" in report

if __name__ == '__main__':
    # Load a sample config for testing
    test_config = {
                'exp': 'test_exp',
                'run': 1,
            }

    # Save test data to a file
#    data = np.load('data.npz')['data']
#    np.savez('test_data.npz', data=data)
    test_config['make_histogram']['input_file'] = 'data.npz'

    # Run LoadData 
    load_data = LoadData(test_config)
    load_data.load_data()
    data_file = load_data.save_data()
    load_data.summarize()

    # Run MakeHistogram
    histogram_maker = MakeHistogram(test_config)
    histogram_maker.load_data()
    histogram_maker.generate_histograms()
    summary = histogram_maker.summarize()
    hfile = histogram_maker.save_histograms()
    histogram_maker.save_histogram_image()
    print(summary)

    # Run MeasureEMD and save the output
    emd_file, null_dist_file = MeasureEMD(test_config).run(hfile)

    # Run CalculatePValues and save the output
    p_values_file = CalculatePValues(test_config).run(emd_file, null_dist_file)

    # Run BuildPumpProbeMasks
    signal_mask_file, bg_mask_file = BuildPumpProbeMasks(test_config).run(p_values_file, hfile)

    try:
        test_pump_probe_analysis(test_config, 'test_data.npz', signal_mask_file, bg_mask_file)
        print("PumpProbeAnalysis test passed")
    except AssertionError as e:
        print(f"PumpProbeAnalysis test failed: {str(e)}")

#if __name__ == '__main__':
#    # TODO test data loader task, but this will only work on s3df
#    # if running in a different environment, download sample data instead
#    data = np.load('data.npz')['arr_0']
#    output_dir = '.'
#    setup_config = {
#        'exp': 'test_exp',
#        'run': 1,
#    }
#
#    hfile, summary = test_make_histogram(data, output_dir, setup_config)
#    print(summary)
#
#    emd_file, null_dist_file = test_measure_emd(hfile, output_dir, setup_config)
#    p_values_file = test_calculate_p_values(emd_file, null_dist_file, output_dir, setup_config)
#    test_build_pump_probe_masks(p_values_file, hfile, output_dir, setup_config)
