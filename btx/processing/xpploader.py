import numpy as np
from pathlib import Path
import tables
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from typing import List

#exp = 'xppx1003221'  
#h5dir = Path('/sdf/data/lcls/ds/xpp/xppx1003221/hdf5/smalldata/')

def projection_fit_func(x, a, c, w, k1, k0):
    return gaus(x, a, c, w) + k1*x + k0

def gaus(x, area, center, width):
    return abs(area) * np.exp(-np.power(x - center, 2.) / (2 * np.power(width, 2.))) / (width * np.sqrt(2 * np.pi))

def SMD_Loader(Run_Number: int) -> tables.File:
    """Load the Small Data"""
    fname = f'{exp}_Run{Run_Number:04d}.h5'
    fname = h5dir / fname
    return tables.open_file(fname).root

def EnergyFilter(rr: tables.File, Energy_Filter: List[float], ROI: List[int]) -> np.ndarray:
    """Threshold the detector images"""
    E0, dE = Energy_Filter
    thresholds = [E0-dE, E0+dE, 2*E0-dE, 2*E0+dE, 3*E0-dE, 3*E0+dE]
    
    imgs_cleaned = rr.jungfrau1M.ROI_0_area[:, ROI[0]:ROI[1], ROI[2]:ROI[3]]
    for i in range(0, len(thresholds), 2):
        imgs_cleaned[(imgs_cleaned > thresholds[i]) & (imgs_cleaned < thresholds[i+1])] = 0
        
    return imgs_cleaned

def fit_LS(func, x, y, initial_guess):
    """Fit the curve"""
    popt, pcov = curve_fit(func, x, y, p0=initial_guess, maxfev=1000000)
    popt, pcov = curve_fit(func, x, y, p0=popt, maxfev=1000000)
    perr = np.sqrt(np.diag(pcov))
    return np.stack((popt, perr), axis=1)
        
def plot_ipm_pos(I0_x, I0_y, arg, IPM_pos_Filter, I0_x_mean, I0_y_mean):
    arg_I0_x = (I0_x < (I0_x_mean + IPM_pos_Filter[0])) & (I0_x > (I0_x_mean - IPM_pos_Filter[0]))
    arg_I0_y = (I0_y < (I0_y_mean + IPM_pos_Filter[1])) & (I0_y > (I0_y_mean - IPM_pos_Filter[1]))
    plt.title('Beam location on BPM...')
    plt.hist2d(I0_x[arg], I0_y[arg], bins=(100,100), norm=LogNorm(), cmap='jet')
    plt.xlabel('ipm2_posx (percentage)')
    plt.ylabel('ipm2_posy (percentage)')
    plt.axvline(I0_x_mean - IPM_pos_Filter[0], color='k')
    plt.axvline(I0_x_mean + IPM_pos_Filter[0], color='k') 
    plt.axhline(I0_y_mean + IPM_pos_Filter[1], color='r')
    plt.axhline(I0_y_mean - IPM_pos_Filter[1], color='r')
    plt.colorbar() 
    plt.minorticks_on()
    plt.show()

def plot_tt_amplidue(amplitude, threshold):
    plt.hist(amplitude, bins=100) 
    plt.xlabel('TimeTool amplitude')
    plt.ylabel('Number of appearances')
    plt.axvline(threshold, color='k')
    plt.minorticks_on()
    plt.yscale('log') 
    plt.show()

def delay_bin(delay: np.ndarray, delay_raw: np.ndarray, Time_bin: float, arg_delay_nan: np.ndarray) -> np.ndarray:
    delay_min = np.floor(delay_raw[~arg_delay_nan].min()) 
    delay_max = np.ceil(delay_raw[~arg_delay_nan].max())
    bins = np.arange(delay_min, delay_max + Time_bin, Time_bin) 
    binned_indices = np.digitize(delay, bins)
    return np.array([bins[idx-1] if idx > 0 else bins[0] for idx in binned_indices])

def imgs_grouping(delay, imgs, I0, mask, arg_delay_nan, arg_I0, arg_I0_x, arg_I0_y, arg_laser_on, arg_laser_off, arg_tt_amplidude, TimeTool, ROI):
    delay_output = np.sort(np.unique(delay[~arg_delay_nan]))
    ims_group_on, ims_group_off, scan_motor = [], [], []
    
    for delay_val in delay_output:
        if TimeTool[0] == 0:
            idx_on = np.where((arg_I0) & (delay == delay_val) & (arg_I0_x) & (arg_I0_y) & (arg_laser_on))[0] 
            idx_off = np.where((arg_I0) & (delay == delay_val) & (arg_I0_x) & (arg_I0_y) & (arg_laser_off))[0]
        elif TimeTool[0] == 1:
            idx_on = np.where((arg_I0) & (delay == delay_val) & (arg_I0_x) & (arg_I0_y) & (arg_laser_on) & (arg_tt_amplidude))[0]
            idx_off = np.where((arg_I0) & (delay == delay_val) & (arg_I0_x) & (arg_I0_y) & (arg_laser_off))[0]
            
        if len(idx_on) > 20 and len(idx_off) > 20:
            print(f'Working on the data of delay {delay_val:.2f} ps...')
            selected_imgs_on = (imgs[idx_on] / I0[idx_on, np.newaxis, np.newaxis]) * mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            selected_imgs_off = (imgs[idx_off] / I0[idx_off, np.newaxis, np.newaxis]) * mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            ims_group_on.append(selected_imgs_on)
            ims_group_off.append(selected_imgs_off)
            print(f'Number of laser on and off events after filtering are {len(idx_on):d}/{len(idx_off):d}.')
            scan_motor.append(delay_val)
            
    ims_group_on = np.concatenate(ims_group_on, axis=0)
    ims_group_off = np.concatenate(ims_group_off, axis=0)
    
    return np.array(scan_motor), ims_group_on, ims_group_off

def CDW_PP(Run_Number: int, ROI: List[int], 
           Energy_Filter: List[float] = [8.8, 5], I0_Threshold: float = 200, 
           IPM_pos_Filter: List[float] = [0.2, 0.5], Time_bin: float = 2, TimeTool: List[float] = [0., 0.005]):
    """Visualize the pump/probe CDW signal"""
    rr = SMD_Loader(Run_Number)
    
    idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0,0]
    mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,1], 
                                                    rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,1]]

    I0 = rr.ipm2.sum[:]
    arg_I0 = I0 >= I0_Threshold 
    
    I0_x, I0_y = rr.ipm2.xpos[:], rr.ipm2.ypos[:]
    arg = (np.abs(I0_x) < 2) & (np.abs(I0_y) < 3)  
    I0_x_mean, I0_y_mean = I0_x[arg].mean(), I0_y[arg].mean()
    arg_I0_x = (I0_x < I0_x_mean + IPM_pos_Filter[0]) & (I0_x > I0_x_mean - IPM_pos_Filter[0])
    arg_I0_y = (I0_y < I0_y_mean + IPM_pos_Filter[1]) & (I0_y > I0_y_mean - IPM_pos_Filter[1])

    plot_ipm_pos(I0_x, I0_y, arg, IPM_pos_Filter, I0_x_mean, I0_y_mean)  

    tt_arg = TimeTool[0]
    delay = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS) * tt_arg
    arg_delay_nan = np.isnan(delay)
    delay = delay_bin(delay, np.array(rr.enc.lasDelay), Time_bin, arg_delay_nan)

    arg_tt_amplidude = np.array(rr.tt.AMPL) > TimeTool[1] 
    
    if tt_arg == 1:
        plot_tt_amplidue(np.array(rr.tt.AMPL), TimeTool[1])
        
    arg_laser_on = np.array(rr.evr.code_90) == 1
    arg_laser_off = np.array(rr.evr.code_91) == 1

    imgs = EnergyFilter(rr, Energy_Filter, ROI)

    delay, imgs_on, imgs_off = imgs_grouping(delay, imgs, I0, mask, arg_delay_nan, arg_I0, arg_I0_x, arg_I0_y, arg_laser_on, arg_laser_off, arg_tt_amplidude, TimeTool, ROI)

    return delay, imgs_on, imgs_off

#def get_imgs_thresh(run_number, experiment_number, roi,
#                    energy_filter=[8.8, 5], i0_threshold=200, 
#                    ipm_pos_filter=[0.2, 0.5], time_bin=2,
#                    time_tool=[0., 0.005]):
#    
#    global exp, h5dir
#    exp = experiment_number
#    h5dir = Path(f'/sdf/data/lcls/ds/xpp/{exp}/hdf5/smalldata/')
#    
#    delay, imgs_on, imgs_off = CDW_PP(run_number, roi, energy_filter, i0_threshold, ipm_pos_filter, time_bin, time_tool)
#    imgs_thresh = imgs_off
#    
#    print(f"Debug: imgs_thresh shape: {imgs_thresh.shape}")
#    
#    return imgs_thresh

def get_imgs_thresh(run_number, experiment_number, roi,
                    energy_filter=[8.8, 5], i0_threshold=200,
                    ipm_pos_filter=[0.2, 0.5], time_bin=2,
                    time_tool=[0., 0.005]):
    global exp, h5dir
    exp = experiment_number
    h5dir = Path(f'/sdf/data/lcls/ds/xpp/{exp}/hdf5/smalldata/')

    rr = SMD_Loader(run_number)

    idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0, 0]
    mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1, 0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1, 1],
                                                    rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2, 0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2, 1]]

    I0 = rr.ipm2.sum[:]
    laser_on_mask = np.array(rr.evr.code_90) == 1
    laser_off_mask = np.array(rr.evr.code_91) == 1

    imgs = EnergyFilter(rr, energy_filter, roi)
    roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi
    imgs_thresh = imgs * mask[roi_x_start:roi_x_end, roi_y_start:roi_y_end]

    tt_arg = time_tool[0]
    laser_delays = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS) * tt_arg

    return imgs_thresh, I0, laser_delays, laser_on_mask, laser_off_mask
