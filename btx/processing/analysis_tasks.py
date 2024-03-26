import numpy as np
from pvalues import  calculate_p_values as _calculate_p_values
from histogram_analysis import identify_roi_connected_cluster, create_background_mask

# TODO this is currently not being used
from pvalues import calculate_emd_values
def calculate_emd(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end, average_histogram = None):
    """
    Calculates Earth Mover's Distance (EMD) values for an array of histograms.
    
    Args:
        histograms (ndarray): 3D array of histograms with shape (bins, rows, cols).
        roi_x_start (int): Start index of the ROI along the x-axis.
        roi_x_end (int): End index of the ROI along the x-axis.
        roi_y_start (int): Start index of the ROI along the y-axis.
        roi_y_end (int): End index of the ROI along the y-axis.
        average_histogram (ndarray, optional): Average histogram to use. If not provided, it will be calculated from the ROI.
        
    Returns:
        ndarray: 2D array of EMD values with shape (rows, cols).
    """
    if histograms.ndim != 3:
        raise ValueError(f"Expected 3D histograms array, got {histograms.ndim}D")
    if not (0 <= roi_x_start < roi_x_end <= histograms.shape[1] and 
            0 <= roi_y_start < roi_y_end <= histograms.shape[2]):
        raise ValueError(f"Invalid ROI coordinates for histogram of shape {histograms.shape}")
        
    if average_histogram is None:
        average_histogram = np.mean(histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end], axis=(1,2))
    
    return calculate_emd_values(histograms, average_histogram)
    

def calculate_p_values(emd_values, null_distribution):
    """
    Calculates p-values from EMD values and a null distribution.
    
    Args:
        emd_values (ndarray): 2D array of EMD values with shape (rows, cols).
        null_distribution (ndarray): 1D array of null distribution EMD values.
        
    Returns:
        ndarray: 2D array of p-values with shape (rows, cols).
    """
    if emd_values.ndim != 2:
        raise ValueError(f"Expected 2D EMD values array, got {emd_values.ndim}D")
    if null_distribution.ndim != 1:
        raise ValueError(f"Expected 1D null distribution array, got {null_distribution.ndim}D")
    
    return _calculate_p_values(emd_values, null_distribution)


def generate_signal_mask(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """
    Generates a binary signal mask from p-values and a threshold.
    
    Args:
        p_values (ndarray): 2D array of p-values with shape (rows, cols).
        threshold (float): P-value threshold for considering a pixel as signal.
        roi_x_start (int): Start index of the ROI along the x-axis.
        roi_x_end (int): End index of the ROI along the x-axis. 
        roi_y_start (int): Start index of the ROI along the y-axis.
        roi_y_end (int): End index of the ROI along the y-axis.
        
    Returns:
        ndarray: 2D binary signal mask with shape (rows, cols).
    """
    if p_values.ndim != 2:
        raise ValueError(f"Expected 2D p-values array, got {p_values.ndim}D") 
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
    labeled_array, roi_connected_cluster = identify_roi_connected_cluster(
            p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end)

    return roi_connected_cluster

    
def generate_background_mask(signal_mask, background_mask_multiple, thickness):
    """
    Generates a background mask from a signal mask.

    Args:
        signal_mask (ndarray): 2D binary array indicating signal pixels.
        background_mask_multiple (float): Multiple of the number of pixels in the signal mask to include in the background.
        thickness (int): Thickness of the background region around the signal.

    Returns:
        ndarray: 2D binary background mask with shape (rows, cols).
    """
    if signal_mask.ndim != 2 or signal_mask.dtype != bool:
        raise ValueError(f"Expected 2D boolean signal mask, got {signal_mask.ndim}D {signal_mask.dtype}")
    if background_mask_multiple < 0:
        raise ValueError(f"Background mask multiple must be non-negative, got {background_mask_multiple}")
    if thickness < 0:
        raise ValueError(f"Background mask thickness must be non-negative, got {thickness}")

    return create_background_mask(signal_mask, background_mask_multiple, thickness)
