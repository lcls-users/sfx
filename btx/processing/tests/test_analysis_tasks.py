import numpy as np
from analysis_tasks import *

def test_calculate_emd():
    histograms = np.random.rand(10, 100, 100)
    emd = calculate_emd(histograms, 40, 60, 40, 60)
    assert emd.shape == (100, 100)
    assert np.all(emd >= 0)

    avg_hist = np.mean(histograms, axis=(1,2))
    emd2 = calculate_emd(histograms, 40, 60, 40, 60, average_histogram=avg_hist) 
    assert np.allclose(emd, emd2)

    try:
        calculate_emd(histograms[:,:50,:50], 40, 60, 40, 60)
        assert False, "Expected ValueError for ROI outside histogram" 
    except ValueError:
        pass

def test_generate_signal_mask():
    p_values = np.random.rand(100, 100)
    mask = generate_signal_mask(p_values, 0.1, 40, 60, 40, 60) 
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    assert 0 < np.sum(mask) < 100*100

    try:
        generate_signal_mask(p_values, 1.1, 40, 60, 40, 60)
        assert False, "Expected ValueError for threshold > 1"
    except ValueError:
        pass
