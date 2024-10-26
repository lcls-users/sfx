from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

@dataclass
class LoadDataInput:
    """Input data structure for LoadData task."""
    config: Dict[str, Any]
    data: Optional[np.ndarray] = None  # Optional (frames, rows, cols) array
    I0: Optional[np.ndarray] = None  # Optional (frames,) array
    laser_delays: Optional[np.ndarray] = None  # Optional (frames,) array
    laser_on_mask: Optional[np.ndarray] = None  # Optional (frames,) boolean array
    laser_off_mask: Optional[np.ndarray] = None  # Optional (frames,) boolean array

@dataclass
class LoadDataOutput:
    """Output data structure for LoadData task."""
    data: np.ndarray  # (frames, rows, cols)
    I0: np.ndarray  # (frames,)
    laser_delays: np.ndarray  # (frames,)
    laser_on_mask: np.ndarray  # (frames,)
    laser_off_mask: np.ndarray  # (frames,)
    binned_delays: np.ndarray  # (frames,)

@dataclass
class MakeHistogramInput:
    """Input for MakeHistogram task."""
    config: Dict[str, Any]
    load_data_output: LoadDataOutput

@dataclass
class MakeHistogramOutput:
    """Output from MakeHistogram task."""
    histograms: np.ndarray  # 3D array (bins, rows, cols)
    bin_edges: np.ndarray   # 1D array of bin edges
    bin_centers: np.ndarray # 1D array of bin centers

@dataclass
class MeasureEMDInput:
    """Input for MeasureEMD task."""
    config: Dict[str, Any]
    histogram_output: MakeHistogramOutput

@dataclass 
class MeasureEMDOutput:
    """Output from MeasureEMD task."""
    emd_values: np.ndarray  # 2D array (rows, cols)
    null_distribution: np.ndarray  # 1D array
    avg_histogram: np.ndarray  # 1D array representing background ROI
    avg_hist_edges: np.ndarray  # 1D array of bin edges for plotting

@dataclass
class CalculatePValuesInput:
    """Input for CalculatePValues task."""
    config: Dict[str, Any]
    emd_output: MeasureEMDOutput

@dataclass
class CalculatePValuesOutput:
    """Output from CalculatePValues task."""
    p_values: np.ndarray  # 2D array (rows, cols)
    log_p_values: np.ndarray  # 2D array -log10(p_values)
    significance_threshold: float  # Configured significance threshold

@dataclass
class BuildPumpProbeMasksInput:
    """Input for BuildPumpProbeMasks task."""
    config: Dict[str, Any]
    histogram_output: MakeHistogramOutput
    p_values_output: CalculatePValuesOutput

@dataclass
class SignalMaskStages:
    """Intermediate stages of signal mask generation."""
    initial: np.ndarray  # After thresholding
    roi_masked: np.ndarray  # After ROI masking
    filtered: np.ndarray  # After cluster filtering
    final: np.ndarray  # After infilling

@dataclass
class BuildPumpProbeMasksOutput:
    """Output from BuildPumpProbeMasks task."""
    signal_mask: np.ndarray  # 2D boolean array
    background_mask: np.ndarray  # 2D boolean array
    intermediate_masks: SignalMaskStages  # Intermediate results for debugging

@dataclass
class PumpProbeAnalysisInput:
    """Input for PumpProbeAnalysis task."""
    config: Dict[str, Any]
    load_data_output: LoadDataOutput
    masks_output: BuildPumpProbeMasksOutput

@dataclass
class DelayGroup:
    """Data for a single delay timepoint."""
    delay: float  # Delay value in ps
    on_frames: np.ndarray  # (n_frames, rows, cols) for laser-on frames
    off_frames: np.ndarray  # (n_frames, rows, cols) for laser-off frames
    on_I0: np.ndarray  # (n_frames,) I0 values for laser-on frames
    off_I0: np.ndarray  # (n_frames,) I0 values for laser-off frames

@dataclass
class PumpProbeAnalysisOutput:
    """Output from PumpProbeAnalysis task."""
    delays: np.ndarray  # 1D array of delay values
    signals_on: np.ndarray  # 1D array of on signals
    signals_off: np.ndarray  # 1D array of off signals
    std_devs_on: np.ndarray  # 1D array of on standard deviations
    std_devs_off: np.ndarray  # 1D array of off standard deviations
    p_values: np.ndarray  # 1D array of p-values
    log_p_values: np.ndarray  # 1D array of -log10(p_values)
    mean_I0_on: float  # Mean I0 value for laser-on frames
    mean_I0_off: float  # Mean I0 value for laser-off frames
    n_frames_per_delay: Dict[float, Tuple[int, int]]  # Dict mapping delay to (n_on, n_off)
