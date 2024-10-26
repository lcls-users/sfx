# extract_h5_data.py
import h5py
import numpy as np
from pathlib import Path

def extract_data_from_h5(filename: str, output_dir: str) -> None:
    """
    Extract data from h5 file and save to npz.
    
    Args:
        filename: Path to h5 file
        output_dir: Directory to save npz file
    """
    # Path mappings in h5 file
    data_paths = {
        'scanvar': 'enc/lasDelay2',    # Delay values
        'i0': 'ipm2/sum',              # I0 intensity
        'roi0': 'jungfrau1M/ROI_0_area',  # Raw frames
        'roi0_mask': 'UserDataCfg/jungfrau1M/ROI_0__ROI_0_mask',
        'xon': 'lightStatus/xray',     # X-ray status
        'lon': 'lightStatus/laser',    # Laser status
        'tt_amp': 'tt/AMPL',           # Time tool amplitude
        'xpos': 'ipm2/xpos',           # Beam position
        'ypos': 'ipm2/ypos'
    }
    
    # Define filters
    filters = {
        'i0': [1500, 20000],           # Intensity filter
        'xpos': [-0.25, 0.45],         # X position
        'ypos': [-0.6, 0.8],           # Y position
        'tt_amp': [0.015, np.inf],     # Time tool amplitude
    }
    
    print("Opening h5 file...")
    with h5py.File(filename, 'r') as h5:
        # 1. Get base masks
        print("Creating base masks...")
        xray_on = h5[data_paths['xon']][:]
        laser = h5[data_paths['lon']][:]
        
        # Initial masks
        filt = xray_on  # Start with x-ray filter
        laser_on_mask = np.logical_and(laser, filt)
        laser_off_mask = np.logical_and(np.logical_not(laser), filt)
        
        # 2. Apply additional filters
        print("Applying filters...")
        for key, (min_val, max_val) in filters.items():
            data = h5[data_paths[key]][:]
            value_filter = np.logical_and(data > min_val, data < max_val)
            
            # Apply to both masks
            laser_on_mask = np.logical_and(laser_on_mask, value_filter)
            if 'tt' not in key:  # Don't apply time tool filters to laser off
                laser_off_mask = np.logical_and(laser_off_mask, value_filter)
        
        print(f"After filtering: {np.sum(laser_on_mask)} laser-on shots, {np.sum(laser_off_mask)} laser-off shots")
        
        # 3. Get delay values
        print("Getting delay values...")
        delays = h5[data_paths['scanvar']][:]
        
        # 4. Get I0 values
        print("Getting I0 values...")
        i0 = h5[data_paths['i0']][:]
        
        # 5. Get detector frames
        print("Loading detector frames...")
        roi0_mask = h5[data_paths['roi0_mask']][0]
        frames = h5[data_paths['roi0']][:] * np.logical_not(roi0_mask)
        
        # Save to npz
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        run_number = Path(filename).stem.split('Run')[1][:4]  # Extract run number
        save_path = output_path / f'run{run_number}_extracted.npz'
        
        print(f"Saving data to {save_path}")
        np.savez(
            save_path,
            frames=frames,
            delays=delays,
            I0=i0,
            laser_on_mask=laser_on_mask,
            laser_off_mask=laser_off_mask,
            # Save filter values for reference
            filter_values={str(k): v for k, v in filters.items()}
        )
        
        print("Extraction complete!")
        return save_path

filename = "xppl1030522_Run0190.h5"
output_dir = "processed_data"
extract_data_from_h5(filename, output_dir)
