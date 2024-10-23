import numpy as np
from typing import Tuple

def generate_synthetic_frames(
    num_frames: int,
    rows: int,
    cols: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for testing LoadData task.
    
    Args:
        num_frames: Number of frames to generate
        rows: Number of rows per frame
        cols: Number of columns per frame
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - data: (num_frames, rows, cols) array of frame data
        - I0: (num_frames,) array of I0 values
        - delays: (num_frames,) array of delay values
        - on_mask: (num_frames,) boolean array for laser on
        - off_mask: (num_frames,) boolean array for laser off
    """
    np.random.seed(seed)
    
    # Create base pattern for frames
    x, y = np.meshgrid(np.linspace(-5, 5, rows), np.linspace(-5, 5, cols))
    base_pattern = np.exp(-(x**2 + y**2)/2)
    
    # Generate frames with noise and varying intensity
    frames = []
    for _ in range(num_frames):
        intensity = np.random.normal(1.0, 0.1)
        noise = np.random.normal(0, 0.05, (rows, cols))
        frame = intensity * base_pattern + noise
        frames.append(frame)
    
    data = np.array(frames)
    
    # Generate other arrays
    I0 = np.random.normal(1000, 100, num_frames)
    delays = np.linspace(-10, 10, num_frames) + np.random.normal(0, 0.1, num_frames)
    on_mask = np.zeros(num_frames, dtype=bool)
    on_mask[::2] = True
    off_mask = ~on_mask
    
    return data, I0, delays, on_mask, off_mask
