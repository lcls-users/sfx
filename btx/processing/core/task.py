# btx/processing/core/task.py
"""Base task definition for pipeline components."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

class Task(ABC):
    """Abstract base class for pipeline tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize task with configuration.
        
        Args:
            config: Configuration dictionary for the task
            
        Raises:
            ValueError: If configuration validation fails
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate task configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Run the task with given input data.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Task output data
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If task execution fails
        """
        pass
    
    def plot_diagnostics(self, output: Any, save_dir: Optional[Path] = None) -> None:
        """Generate diagnostic plots for task output.
        
        Args:
            output: Task output data
            save_dir: Directory to save plots. If None, display plots interactively.
        """
        pass
