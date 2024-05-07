import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm
import tables
from pathlib import Path

def get_imgs_thresh(run_number, experiment_number, roi,
                    energy_filter=[8.8, 5], i0_threshold=200, 
                    ipm_pos_filter=[0.2, 0.5], time_bin=2,
                    time_tool=[0., 0.005]):

    delay, imgs_on, imgs_off = get_grouped_images(run_number, roi, energy_filter, 
                                                  i0_threshold, ipm_pos_filter, 
                                                  time_bin, time_tool, experiment_number)
    
    imgs_thresh = np.concatenate(imgs_off, axis=0)
    
    return imgs_thresh


def get_grouped_images(run_number, roi, energy_filter, i0_threshold,
                       ipm_pos_filter, time_bin, time_tool, experiment_number):
    
    delay, imgs_on, imgs_off = CDW_PP(run_number, roi, energy_filter, i0_threshold, 
                                      ipm_pos_filter, time_bin, time_tool, experiment_number)
    
    return delay, imgs_on, imgs_off


def CDW_PP(run_number, roi, energy_filter, i0_threshold, 
           ipm_pos_filter, time_bin, time_tool, experiment_number):
    
    exp_path = Path(f'/sdf/data/lcls/ds/xpp/{experiment_number}/hdf5/smalldata/')
    
    rr = SMD_Loader(run_number, exp_path, experiment_number)
    
    I0 = rr.ipm2.sum[:]
    arg_I0 = (I0 >= i0_threshold)
    
    I0_x = rr.ipm2.xpos[:]
    I0_y = rr.ipm2.ypos[:]
    arg = (abs(I0_x)<2.) & (abs(I0_y)<3.) 
    I0_x_mean, I0_y_mean = I0_x[arg].mean(), I0_y[arg].mean()
    arg_I0_x = (I0_x < (I0_x_mean + ipm_pos_filter[0])) & (I0_x > (I0_x_mean - ipm_pos_filter[0]))
    arg_I0_y = (I0_y < (I0_y_mean + ipm_pos_filter[1])) & (I0_y > (I0_y_mean - ipm_pos_filter[1]))

    plot_ipm_pos(I0_x, I0_y, arg, ipm_pos_filter, I0_x_mean, I0_y_mean)
    
    tt_arg = time_tool[0]
    delay = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS)*tt_arg
    arg_delay_nan = np.isnan(delay)
    delay = delay_bin(delay, np.array(rr.enc.lasDelay), time_bin, arg_delay_nan)

    arg_tt_amplitude = (np.array(rr.tt.AMPL) > time_tool[1])
    
    if tt_arg == 1.:
        plot_tt_amplitude(np.array(rr.tt.AMPL), time_tool[1]) 
        
    arg_laser_on = (np.array(rr.evr.code_90) == 1.)
    arg_laser_off = (np.array(rr.evr.code_91) == 1.)
    
    imgs = EnergyFilter(rr, energy_filter, roi)
    
    mask = rr.UserDataCfg.jungfrau1M.mask[rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0,0]][roi[0]:roi[1], roi[2]:roi[3]]
    
    delay, imgs_on, imgs_off = imgs_grouping(delay, imgs, I0, mask, arg_delay_nan, arg_I0, 
                                             arg_I0_x, arg_I0_y, arg_laser_on, arg_laser_off,
                                             arg_tt_amplitude, time_tool, roi)
    
    return delay, imgs_on, imgs_off


def SMD_Loader(run_number, exp_path, experiment_number):
    fname = f'{experiment_number}_Run{run_number:04d}.h5'
    fname = exp_path / fname
    rr = tables.open_file(fname).root
    return rr


def EnergyFilter(rr, energy_filter, roi):
    E0, dE = energy_filter[0], energy_filter[1]
    thresh_1, thresh_2 = E0-dE, E0+dE 
    thresh_3, thresh_4 = 2*E0-dE, 2*E0+dE
    thresh_5, thresh_6 = 3*E0-dE, 3*E0+dE

    imgs_cleaned = rr.jungfrau1M.ROI_0_area[:, roi[0]:roi[1], roi[2]:roi[3]]
    imgs_cleaned[(imgs_cleaned<thresh_1) 
                 |((imgs_cleaned>thresh_2) & (imgs_cleaned<thresh_3))
                 |((imgs_cleaned>thresh_4) & (imgs_cleaned<thresh_5)) 
                 |(imgs_cleaned>thresh_6)] = 0
    
    return imgs_cleaned


def plot_ipm_pos(I0_x, I0_y, arg, ipm_pos_filter, I0_x_mean, I0_y_mean):
    arg_I0_x = (I0_x < (I0_x_mean + ipm_pos_filter[0])) & (I0_x > (I0_x_mean - ipm_pos_filter[0]))
    arg_I0_y = (I0_y < (I0_y_mean + ipm_pos_filter[1])) & (I0_y > (I0_y_mean - ipm_pos_filter[1]))
    plt.figure()
    plt.title('Beam location on BPM...')
    plt.hist2d(I0_x[arg], I0_y[arg], bins=(100,100), norm=LogNorm(), cmap='jet')
    plt.xlabel('ipm2_posx (percentage)')
    plt.ylabel('ipm2_posy (percentage)')
    plt.axvline(I0_x_mean - ipm_pos_filter[0], color='k')
    plt.axvline(I0_x_mean + ipm_pos_filter[0], color='k')
    plt.axhline(I0_y_mean + ipm_pos_filter[1], color='r') 
    plt.axhline(I0_y_mean - ipm_pos_filter[1], color='r')
    plt.colorbar()
    plt.minorticks_on()
    plt.show()
    
    
def plot_tt_amplitude(amplitude, threshold):
    plt.figure()
    plt.hist(amplitude, bins=100) 
    plt.xlabel('TimeTool amplitude')
    plt.ylabel('Number of appearances')
    plt.axvline(threshold, color='k')
    plt.minorticks_on()
    plt.yscale('log') 
    plt.show()
    
    
def delay_bin(delay, delay_raw, time_bin, arg_delay_nan):
    time_bin = float(time_bin)
    delay_min = np.floor(delay_raw[arg_delay_nan == False].min())
    delay_max = np.ceil(delay_raw[arg_delay_nan == False].max())
    bins = np.arange(delay_min, delay_max + time_bin, time_bin)
    binned_indices = np.digitize(delay, bins)
    binned_delays = np.array([bins[idx-1] if idx > 0 else bins[0] for idx in binned_indices])
    return binned_delays


def imgs_grouping(delay, imgs, I0, mask, arg_delay_nan, arg_I0, arg_I0_x, arg_I0_y, 
                  arg_laser_on, arg_laser_off, arg_tt_amplitude, time_tool, roi):
    
    delay_output = list(set(delay[arg_delay_nan == False]))
    delay_output = np.sort(np.array(delay_output))
    ims_group_on, ims_group_off, scan_motor = [], [], []
    
    for i in range(len(delay_output)):
        if time_tool[0] == 0:
            idx_on = np.where((arg_I0 == True) & (delay == delay_output[i]) & (arg_I0_x == True) & (arg_I0_y == True) & (arg_laser_on == True))[0]
            idx_off = np.where((arg_I0 == True) & (delay == delay_output[i]) & (arg_I0_x == True) & (arg_I0_y == True) & (arg_laser_off == True))[0]
        elif time_tool[0] == 1.:
            idx_on = np.where((arg_I0 == True) & (delay == delay_output[i]) & (arg_I0_x == True) & (arg_I0_y == True) & (arg_laser_on == True) & (arg_tt_amplitude == True))[0]
            idx_off = np.where((arg_I0 == True) & (delay == delay_output[i]) & (arg_I0_x == True) & (arg_I0_y == True) & (arg_laser_off == True))[0]
        
        if (len(idx_on) > 20) & (len(idx_off) > 20):
            ims_group_on.append((imgs[idx_on].mean(axis=0) / I0[idx_on].mean()) * mask)
            ims_group_off.append((imgs[idx_off].mean(axis=0) / I0[idx_off].mean()) * mask)
            scan_motor.append(delay_output[i])
    
    ims_group_on = np.array(ims_group_on)
    ims_group_off = np.array(ims_group_off) 
    scan_motor = np.array(scan_motor)
    
    return scan_motor, ims_group_on, ims_group_off
