# tests/functional/utils.py

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

def generate_test_data(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float64,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic test data.
    
    Args:
        shape: Shape of array to generate
        dtype: Data type of array
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Synthetic data array
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate base pattern
    center = np.array([s/2 for s in shape])
    indices = np.indices(shape)
    r2 = sum((ind - c)**2 for ind, c in zip(indices, center))
    base = np.exp(-r2 / (2 * max(shape)**2))
    
    # Add noise
    noise = np.random.normal(0, noise_level, shape)
    
    return (base + noise).astype(dtype)

def plot_array_comparison(
    arrays: dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    suptitle: Optional[str] = None,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Create comparison plot of multiple arrays.
    
    Args:
        arrays: Dictionary mapping names to arrays
        save_path: Path to save plot (if None, display)
        suptitle: Super title for whole figure
        **kwargs: Additional kwargs for imshow
        
    Returns:
        Figure object if save_path is None
    """
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
        
    for ax, (name, arr) in zip(axes, arrays.items()):
        im = ax.imshow(arr, **kwargs)
        plt.colorbar(im, ax=ax)
        ax.set_title(name)
        
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig

def verify_array_properties(
    arr: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[np.dtype] = None,
    bounds: Optional[Tuple[float, float]] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> bool:
    """
    Verify array properties.
    
    Args:
        arr: Array to verify
        expected_shape: Expected shape (if None, not checked)
        expected_dtype: Expected dtype (if None, not checked)
        bounds: Expected (min, max) bounds (if None, not checked)
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether infinite values are allowed
        
    Returns:
        True if all checks pass
        
    Raises:
        AssertionError: If any check fails
    """
    if expected_shape is not None:
        assert arr.shape == expected_shape, \
            f"Wrong shape: expected {expected_shape}, got {arr.shape}"
            
    if expected_dtype is not None:
        assert arr.dtype == expected_dtype, \
            f"Wrong dtype: expected {expected_dtype}, got {arr.dtype}"
            
    if bounds is not None:
        assert arr.min() >= bounds[0] and arr.max() <= bounds[1], \
            f"Values outside bounds {bounds}: min={arr.min()}, max={arr.max()}"
            
    if not allow_nan:
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        
    if not allow_inf:
        assert not np.any(np.isinf(arr)), "Array contains infinite values"
        
    return True

def visual_check(
    fig: plt.Figure,
    prompt: str = "Does the plot look correct?"
) -> bool:
    """
    Prompt for visual verification of plot.
    
    Args:
        fig: Figure to display
        prompt: Question to ask user
        
    Returns:
        True if user confirms plot looks correct
    """
    plt.show()
    response = input(f"{prompt} (y/n): ")
    plt.close(fig)
    return response.lower() == 'y'
