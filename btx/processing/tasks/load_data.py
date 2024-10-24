from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from btx.processing.btx_types import LoadDataInput, LoadDataOutput

class LoadData:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def run(self, input_data: Optional[LoadDataInput]) -> LoadDataOutput:  # Modified signature
        """Run the data loading and preprocessing.
        
        Args:
            input_data: Optional input data for synthetic testing
            
        Returns:
            LoadDataOutput containing processed data
        """
        if input_data is not None:
            # Use provided synthetic data
            data = input_data.data
            I0 = input_data.I0
            laser_delays = input_data.laser_delays
            laser_on_mask = input_data.laser_on_mask
            laser_off_mask = input_data.laser_off_mask
        else:
            # Load from file using get_imgs_thresh
            try:
                from btx.processing.xpploader import get_imgs_thresh
            except ImportError:
                print("Note: get_imgs_thresh not available, only synthetic data mode supported")
                raise
                
            data, I0, laser_delays, laser_on_mask, laser_off_mask = get_imgs_thresh(
                self.config['setup']['run'],
                self.config['setup']['exp'],
                self.config['load_data']['roi'],
                self.config['load_data'].get('energy_filter', [8.8, 5]),
                self.config['load_data'].get('i0_threshold', 200),
                self.config['load_data'].get('ipm_pos_filter', [0.2, 0.5]),
                self.config['load_data'].get('time_bin', 2),
                self.config['load_data'].get('time_tool', [0., 0.005])
            )
            
        # Calculate binned delays
        binned_delays = self._calculate_binned_delays(laser_delays)
        
        return LoadDataOutput(
            data=data,
            I0=I0,
            laser_delays=laser_delays,
            laser_on_mask=laser_on_mask,
            laser_off_mask=laser_off_mask,
            binned_delays=binned_delays
        )

    def plot_diagnostics(self, output: LoadDataOutput, save_dir: Path) -> None:
        """Generate diagnostic plots for visual validation."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average frame
        avg_frame = np.mean(output.data, axis=0)
        im1 = ax1.imshow(avg_frame, cmap='viridis')
        ax1.set_title('Average Frame')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Frame-to-frame intensity variation
        ax2.plot(np.mean(output.data, axis=(1,2)), label='Mean Intensity')
        ax2.plot(output.I0, label='I0', alpha=0.5)
        ax2.set_title('Intensity Variation')
        ax2.set_xlabel('Frame')
        ax2.legend()
        
        # 3. Delay distribution
        ax3.hist(output.laser_delays[output.laser_on_mask], bins=50, 
                 alpha=0.5, label='Laser On')
        ax3.hist(output.laser_delays[output.laser_off_mask], bins=50,
                 alpha=0.5, label='Laser Off')
        ax3.set_title('Delay Distribution')
        ax3.set_xlabel('Delay (ps)')
        ax3.legend()
        
        # 4. Binned delays vs raw delays
        ax4.scatter(output.laser_delays, output.binned_delays, alpha=0.1)
        ax4.plot([output.laser_delays.min(), output.laser_delays.max()],
                [output.laser_delays.min(), output.laser_delays.max()],
                'r--', label='y=x')
        ax4.set_title('Binned vs Raw Delays')
        ax4.set_xlabel('Raw Delays (ps)')
        ax4.set_ylabel('Binned Delays (ps)')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'load_data_diagnostics.png')
        plt.close()
