# btx/processing/core/adapters.py
"""Task adaptation and registry components."""

from typing import Dict, Any, Type, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

from .task import Task

@dataclass
class TaskAdapter(Task):
    """Simplified adapter to wrap task classes."""
    task_instance: Any
    name: str

    def __init__(self, task_class: Type[Any], config: Dict[str, Any], name: str):
        """Initialize task adapter.
        
        Args:
            task_class: Original task class to wrap
            config: Configuration dictionary
            name: Task name for error reporting
        """
        super().__init__(config)
        self.task_instance = task_class(config)
        self.name = name
    
    def run(self, input_data: Any = None) -> Any:
        """Run the wrapped task."""
        try:
            return self.task_instance.run(input_data)
        except Exception as e:
            raise RuntimeError(f"Task {self.name} failed: {str(e)}") from e
    
    def plot_diagnostics(self, output: Any, save_dir: Optional[Path] = None) -> None:
        """Generate diagnostic plots if supported."""
        if hasattr(self.task_instance, 'plot_diagnostics'):
            self.task_instance.plot_diagnostics(output, save_dir)

class TaskRegistry:
    """Simplified registry of tasks."""
    
    def __init__(self):
        self._registry: Dict[str, Type[Any]] = {}
    
    def register(self, name: str, task_class: Type[Any]) -> None:
        """Register a task class."""
        if name in self._registry:
            raise ValueError(f"Task {name} already registered")
        self._registry[name] = task_class
    
    def create(self, name: str, config: Dict[str, Any]) -> TaskAdapter:
        """Create an adapter instance for a task."""
        if name not in self._registry:
            raise ValueError(f"Task {name} not registered")
        
        return TaskAdapter(
            task_class=self._registry[name],
            config=config,
            name=name
        )

# Global registry instance
registry = TaskRegistry()

def register_task(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a task class.
    
    Args:
        name: Task name
        
    Returns:
        Decorator function
        
    Example:
        @register_task("load_data")
        class LoadData:
            ...
    """
    def decorator(task_class: Type[Any]) -> Type[Any]:
        registry.register(name, task_class)
        return task_class
    return decorator
