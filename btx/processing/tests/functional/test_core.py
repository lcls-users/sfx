# btx/processing/tests/functional/test_core.py

import numpy as np
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional

from btx.processing.core import (
    XPPTask,
    ComposableTask,
    ValidationMixin,
    VisualizationMixin,
    ConfigError,
    DataError
)
from btx.processing.types import MetadataBase

class MockTask(XPPTask[np.ndarray, np.ndarray]):
    """Mock task for testing base functionality"""
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        if 'test_param' not in config:
            raise ConfigError("Missing test_param")
    
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        # Simple processing - add constant from config
        return input_data + self.config['test_param']
    
    def _create_plot(self, output: np.ndarray, **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        im = ax.imshow(output)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{self.__class__.__name__} Output")
        return fig

    def plot(self, output: np.ndarray, save_path: Optional[Path] = None, **kwargs) -> None:
        """Override plot to create directory and use specific filename"""
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            fig = self._create_plot(output, **kwargs)
            fig.savefig(save_path / "output.png")
            plt.close(fig)

class MockTask2(MockTask):
    """Second mock task for testing composition"""
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return input_data * self.config['test_param']

@pytest.fixture
def mock_config():
    """Generate mock configuration"""
    return {'test_param': 1.0}

@pytest.fixture
def test_array():
    """Generate test array data"""
    return np.random.rand(10, 10)

@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for test outputs"""
    return tmp_path_factory.mktemp("test_outputs")

def test_validation_mixin():
    """Test array validation functionality"""
    task = MockTask({'test_param': 1.0})
    
    # Valid array
    valid_arr = np.ones((5, 5))
    task.validate_array(valid_arr, "test", expected_shape=(5, 5))
    
    # Invalid shape
    with pytest.raises(DataError):
        task.validate_array(valid_arr, "test", expected_shape=(3, 3))
        
    # Invalid type
    with pytest.raises(DataError):
        task.validate_array([1, 2, 3], "test")
        
    # NaN/Inf checks
    arr_with_nan = np.array([1.0, np.nan])
    arr_with_inf = np.array([1.0, np.inf])
    
    with pytest.raises(DataError):
        task.validate_array(arr_with_nan, "test", allow_nan=False)
    
    with pytest.raises(DataError):
        task.validate_array(arr_with_inf, "test", allow_inf=False)
        
    # Bounds check
    arr_with_values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(DataError):
        task.validate_array(arr_with_values, "test", bounds=(0.0, 2.0))

def test_visualization_mixin(test_array, tmp_path):
    """Test visualization functionality"""
    task = MockTask({'test_param': 1.0})
    
    # Test saving plot
    plot_path = tmp_path / "test_plot.png"
    task.plot(test_array, save_path=tmp_path)
    assert (tmp_path / "output.png").exists()
    
    # Test plot creation
    fig = task._create_plot(test_array)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_xpp_task(mock_config, test_array, tmp_path):
    """Test XPPTask base functionality"""
    task = MockTask(mock_config)

    # Test basic processing
    result = task.run(test_array)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, test_array + mock_config['test_param'])

    # Test without visualization
    result = task.run(test_array)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, test_array + mock_config['test_param'])

    # Test with visualization but no plot_dir (should raise error)
    with pytest.raises(ValueError):
        task.run(test_array, visualize=True)

    # Test with visualization
    result = task.run(
        test_array,
        visualize=True,
        plot_dir=tmp_path
    )
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, test_array + mock_config['test_param'])
    assert (tmp_path / "output.png").exists()
    
    # Test config validation
    with pytest.raises(ConfigError):
        MockTask({})  # Missing required param

def test_composable_task(mock_config, test_array, tmp_path):
    """Test ComposableTask functionality"""
    task1 = MockTask(mock_config)
    task2 = MockTask2(mock_config)
    
    # Create composite task
    composite = ComposableTask([task1, task2])
    
    # Test basic processing without visualization
    result = composite.run(test_array)
    expected = (test_array + mock_config['test_param']) * mock_config['test_param']
    assert np.allclose(result, expected)
    
    # Test with visualization but no plot_dir (should raise error)
    with pytest.raises(ValueError):
        composite.run(test_array, visualize=True)
    
    # Test with visualization
    result = composite.run(
        test_array,
        visualize=True,
        plot_dir=tmp_path
    )
    assert np.allclose(result, expected)
    
    # Check that visualization files were created
    task1_plot = tmp_path / "task_1_MockTask" / "output.png"
    task2_plot = tmp_path / "task_2_MockTask2" / "output.png"
    assert task1_plot.exists()
    assert task2_plot.exists()

def test_metadata_base():
    """Test MetadataBase functionality"""
    # Test default initialization
    metadata = MetadataBase()
    assert isinstance(metadata.shapes, dict)
    assert isinstance(metadata.timestamp, str)
    assert isinstance(metadata.config_summary, dict)
    
    # Test initialization with values
    shapes = {'data': (10, 10)}
    config = {'param': 1.0}
    metadata = MetadataBase(shapes=shapes, config_summary=config)
    
    assert metadata.shapes == shapes
    assert metadata.config_summary == config
    assert isinstance(metadata.timestamp, str)
    
    # Test modification
    metadata.shapes['new_data'] = (5, 5)
    assert metadata.shapes['new_data'] == (5, 5)
    
    metadata.config_summary['new_param'] = 2.0
    assert metadata.config_summary['new_param'] == 2.0

if __name__ == '__main__':
    pytest.main([__file__])
