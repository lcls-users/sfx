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

    # Check if binned_delays are saved in the output file
    with np.load(output_file) as data:
        assert 'binned_delays' in data, "binned_delays not found in the output file"
        assert data['binned_delays'].shape == data['laser_delays'].shape, "Shape mismatch between binned_delays and laser_delays"

    return output_file

def test_make_histogram(data_file, output_dir, setup_config):
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

    # Create and run the MakeHistogram instance
    make_histogram = MakeHistogram(config)
    make_histogram.load_data()
    make_histogram.generate_histograms()
    make_histogram.summarize()

    # Save the histograms using the save_histograms() method
    histogram_file = make_histogram.save_histograms()

    # Check if the histogram file exists
    assert os.path.exists(histogram_file), f"Histogram file {histogram_file} does not exist"

    # Save the histogram image
    make_histogram.save_histogram_image()

    # Return the path to the histogram file and the summary
    return histogram_file, make_histogram.histogram_summary

def test_make_histogram_from_npz(npz_file, output_dir, setup_config):
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

    # Create and run the MakeHistogram instance
    make_histogram = MakeHistogram(config)
    
    # Load data directly from the npz file
    with np.load(npz_file) as data:
        make_histogram.data = data['data']
    
    make_histogram.generate_histograms()
    make_histogram.summarize()

    # Save the histograms using the save_histograms() method
    histogram_file = make_histogram.save_histograms()

    # Check if the histogram file exists
    assert os.path.exists(histogram_file), f"Histogram file {histogram_file} does not exist"

    # Save the histogram image
    make_histogram.save_histogram_image()

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
    signal_mask_file, bg_mask_file = mask_task.save_masks()

    assert os.path.exists(report_path)
    assert os.path.exists(signal_mask_file)
    assert os.path.exists(bg_mask_file)

    return signal_mask_file, bg_mask_file


def test_pump_probe_analysis(config, data_file, signal_mask_file, bg_mask_file):
    # Load the output data from LoadData and BuildPumpProbeMasks
    data = np.load(data_file, allow_pickle=True)
    signal_mask = np.load(signal_mask_file)
    bg_mask = np.load(bg_mask_file)

    # Initialize and run PumpProbeAnalysis
    analysis = PumpProbeAnalysis(config)
    analysis.run(data['data'], data['binned_delays'], data['laser_on_mask'], data['laser_off_mask'], signal_mask, bg_mask)

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
        assert curves[key].shape == (len(analysis.delays),), f"Unexpected shape for {key} in pump_probe_curves.npz"

    # Check pump_probe_summary.txt
    with open(os.path.join(analysis.output_dir, 'pump_probe_summary.txt'), 'r') as f:
        summary = f.read()
    assert f"Pump-probe analysis for {config['setup']['exp']} run {config['setup']['run']}:" in summary
    assert f"Number of delays: {len(analysis.delays)}" in summary
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
    # TODO unpack setup correctly
    test_config = {
        'exp': 'xppx1003221',
        'run': 195,
        'setup': {
            'exp': 'xppx1003221',
            'run': 195,
            },
        'pump_probe_analysis': {
            'output_dir': 'pump_probe_output',
            # Add other required configuration values for pump-probe analysis
        }
    }

    # Run LoadData
    data_file = test_load_data('loaded_data', test_config)

    # Run MakeHistogram with LoadData output
    hfile_from_load_data, summary_from_load_data = test_make_histogram(data_file, 'histograms_from_load_data', test_config)
    print("MakeHistogram with LoadData output:")
    print(summary_from_load_data)

#    # Run MakeHistogram with pre-existing npz file
#    npz_file = 'data.npz'  # Replace with the actual path to your npz file
#    hfile_from_npz, summary_from_npz = test_make_histogram_from_npz(npz_file, 'histograms_from_npz', test_config)
#    print("MakeHistogram with pre-existing npz file:")
#    print(summary_from_npz)

    # Run MeasureEMD and save the output
    emd_file, null_dist_file = test_measure_emd(hfile_from_load_data, 'emd_output', test_config)

    # Run CalculatePValues and save the output
    p_values_file = test_calculate_p_values(emd_file, null_dist_file, 'pvalues_output', test_config)

    # Run BuildPumpProbeMasks
    signal_mask_file, bg_mask_file = test_build_pump_probe_masks(p_values_file, hfile_from_load_data, 'masks_output', test_config)

    try:
        test_pump_probe_analysis(test_config, data_file, signal_mask_file, bg_mask_file)
        print("PumpProbeAnalysis test passed")
    except AssertionError as e:
        print(f"PumpProbeAnalysis test failed: {str(e)}")

