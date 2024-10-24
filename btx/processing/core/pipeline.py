from typing import Dict, Any, Optional, Set, List, Union
from dataclasses import dataclass, field
import inspect
import time
import traceback
from pathlib import Path
import json
from datetime import datetime
from .task import Task

@dataclass
class TaskResult:
    """Container for individual task execution results."""
    output: Any
    execution_time: float
    memory_usage: float
    start_time: datetime
    end_time: datetime
    success: bool
    error: Optional[str] = None
    traceback: Optional[str] = None

@dataclass
class PipelineResult:
    """Container for pipeline execution results and metadata."""
    results: Dict[str, TaskResult]
    execution_order: List[str]
    start_time: datetime
    end_time: datetime
    config: Dict[str, Any]
    total_execution_time: float
    success: bool = True
    error: Optional[str] = None
    
    def save(self, save_dir: Path) -> None:
        """Save pipeline results to directory.
        
        Args:
            save_dir: Directory to save results
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'execution_order': self.execution_order,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_execution_time': self.total_execution_time,
            'success': self.success,
            'error': self.error
        }
        
        with open(save_dir / 'pipeline_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save task results
        for task_name, result in self.results.items():
            task_dir = save_dir / task_name
            task_dir.mkdir(exist_ok=True)
            
            task_metadata = {
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'success': result.success,
                'error': result.error,
                'traceback': result.traceback
            }
            
            with open(task_dir / 'task_metadata.json', 'w') as f:
                json.dump(task_metadata, f, indent=2)

class Pipeline:
    """Pipeline composition and execution framework."""
    
    def __init__(self, name: str):
        """Initialize pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.execution_order: List[str] = []
        self.results_cache: Dict[str, Any] = {}
        self.diagnostics_dir: Optional[Path] = None
    
    def set_diagnostics_dir(self, path: Union[str, Path]) -> None:
        """Set directory for diagnostic outputs.
        
        Args:
            path: Directory path for diagnostic outputs
        """
        self.diagnostics_dir = Path(path)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    def add_task(self, name: str, task: Task, depends_on: Optional[List[str]] = None) -> None:
        """Add a task to the pipeline with optional dependencies.
        
        Args:
            name: Task name
            task: Task instance
            depends_on: List of task names this task depends on
            
        Raises:
            ValueError: If dependencies are invalid or create cycles
        """
        if name in self.tasks:
            raise ValueError(f"Task {name} already exists in pipeline")
        
        self.tasks[name] = task
        self.dependencies[name] = set(depends_on or [])
        self._validate_dependencies()
        self._compute_execution_order()
    
    def _validate_dependencies(self) -> None:
        """Validate task dependencies and check for cycles.
        
        Raises:
            ValueError: If dependencies are invalid or create cycles
        """
        # Check all dependencies exist
        for task_name, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self.tasks:
                    raise ValueError(f"Task {task_name} depends on non-existent task {dep}")
        
        # Check for cycles
        visited = set()
        path = set()
        
        def visit(task_name: str) -> None:
            if task_name in path:
                cycle = list(path)
                cycle.append(task_name)
                raise ValueError(f"Cyclic dependency detected: {' -> '.join(cycle)}")
            
            if task_name in visited:
                return
            
            visited.add(task_name)
            path.add(task_name)
            
            for dep in self.dependencies[task_name]:
                visit(dep)
            
            path.remove(task_name)
        
        for task_name in self.tasks:
            visit(task_name)
    
    def _compute_execution_order(self) -> None:
        """Compute topological sort of tasks based on dependencies."""
        self.execution_order = []
        visited = set()
        
        def visit(task_name: str) -> None:
            if task_name in visited:
                return
            
            visited.add(task_name)
            for dep in self.dependencies[task_name]:
                visit(dep)
            
            self.execution_order.append(task_name)
        
        for task_name in self.tasks:
            visit(task_name)
    
    def _prepare_task_input(self, task_name: str) -> Any:
        """Prepare input for a task based on its dependencies.
        
        Args:
            task_name: Name of task to prepare input for
            
        Returns:
            Task input data
            
        Raises:
            ValueError: If input preparation fails
        """
        deps = self.dependencies[task_name]
        if not deps:
            return None
        
        # If task has only one dependency, pass that result directly
        if len(deps) == 1:
            dep = next(iter(deps))
            return self.results_cache[dep].output
        
        # For multiple dependencies, construct input based on task's input type hints
        task = self.tasks[task_name]
        sig = inspect.signature(task.run)
        input_type = sig.parameters['input_data'].annotation
        
        if hasattr(input_type, '__annotations__'):
            # Create input object with results from dependencies
            input_args = {}
            for field_name, field_type in input_type.__annotations__.items():
                if field_name == 'config':
                    input_args[field_name] = task.config
                    continue
                
                # Find dependency that produces this type
                for dep in deps:
                    dep_result = self.results_cache[dep].output
                    if isinstance(dep_result, field_type):
                        input_args[field_name] = dep_result
                        break
            
            return input_type(**input_args)
        
        raise ValueError(f"Cannot determine input structure for task {task_name}")
    
    def run(self, initial_input: Optional[Any] = None) -> PipelineResult:
        """Run the pipeline with optional initial input.
        
        Args:
            initial_input: Input data for first task
            
        Returns:
            PipelineResult containing execution results and metadata
            
        Raises:
            RuntimeError: If pipeline execution fails
        """
        self.results_cache.clear()
        pipeline_start = datetime.now()
        pipeline_success = True
        pipeline_error = None
        
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            
            # Prepare input
            try:
                if not self.dependencies[task_name] and task_name == self.execution_order[0]:
                    task_input = initial_input
                else:
                    task_input = self._prepare_task_input(task_name)
            except Exception as e:
                error_msg = f"Failed to prepare input for task {task_name}: {str(e)}"
                self.results_cache[task_name] = TaskResult(
                    output=None,
                    execution_time=0,
                    memory_usage=0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    success=False,
                    error=error_msg,
                    traceback=traceback.format_exc()
                )
                pipeline_success = False
                pipeline_error = error_msg
                break
            
            # Run task
            start_time = datetime.now()
            try:
                result = task.run(task_input)
                
                # Generate diagnostics if directory is set
                if self.diagnostics_dir is not None:
                    task_dir = self.diagnostics_dir / task_name
                    task.plot_diagnostics(result, task_dir)
                
                task_result = TaskResult(
                    output=result,
                    execution_time=time.time() - start_time.timestamp(),
                    memory_usage=0,  # TODO: Implement memory tracking
                    start_time=start_time,
                    end_time=datetime.now(),
                    success=True
                )
                
            except Exception as e:
                error_msg = f"Task {task_name} failed: {str(e)}"
                task_result = TaskResult(
                    output=None,
                    execution_time=time.time() - start_time.timestamp(),
                    memory_usage=0,
                    start_time=start_time,
                    end_time=datetime.now(),
                    success=False,
                    error=error_msg,
                    traceback=traceback.format_exc()
                )
                pipeline_success = False
                pipeline_error = error_msg
            
            self.results_cache[task_name] = task_result
            
            # Stop pipeline if task failed
            if not task_result.success:
                break
        
        pipeline_end = datetime.now()
        total_time = (pipeline_end - pipeline_start).total_seconds()
        
        return PipelineResult(
            results=self.results_cache,
            execution_order=self.execution_order,
            start_time=pipeline_start,
            end_time=pipeline_end,
            total_execution_time=total_time,
            config={name: task.config for name, task in self.tasks.items()},
            success=pipeline_success,
            error=pipeline_error
        )
