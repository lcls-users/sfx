from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass
from pathlib import Path
from .task import Task

@dataclass
class TaskResult:
    """Simplified task result container."""
    output: Any
    success: bool
    error: Optional[str] = None

class Pipeline:
    """Simplified pipeline framework."""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.execution_order: List[str] = []
        self.diagnostics_dir: Optional[Path] = None
    
    def add_task(self, name: str, task: Task, depends_on: Optional[List[str]] = None) -> None:
        """Add a task to the pipeline."""
        if name in self.tasks:
            raise ValueError(f"Task {name} already exists")
        
        self.tasks[name] = task
        self.dependencies[name] = set(depends_on or [])
        self._compute_execution_order()
    
    def _compute_execution_order(self) -> None:
        """Compute topological sort of tasks."""
        self.execution_order = []
        visited = set()
        path = set()
        
        def visit(task_name: str) -> None:
            if task_name in path:
                cycle = list(path)
                cycle.append(task_name)
                raise RuntimeError(f"Cyclic dependency detected: {' -> '.join(cycle)}")
            
            if task_name in visited:
                return
            
            visited.add(task_name)
            path.add(task_name)
            
            for dep in self.dependencies[task_name]:
                if dep not in self.tasks:
                    raise RuntimeError(f"Task {task_name} depends on non-existent task {dep}")
                visit(dep)
            
            path.remove(task_name)
            self.execution_order.append(task_name)
        
        for task_name in self.tasks:
            visit(task_name)
    
    def run(self, initial_input: Optional[Any] = None) -> Dict[str, TaskResult]:
        """Run the pipeline."""
        results = {}
        
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            
            # Prepare input
            if not self.dependencies[task_name] and task_name == self.execution_order[0]:
                task_input = initial_input
            else:
                deps = self.dependencies[task_name]
                if len(deps) == 1:
                    task_input = results[next(iter(deps))].output
                else:
                    # For multiple dependencies, pass dictionary of outputs
                    task_input = {dep: results[dep].output for dep in deps}
            
            # Run task
            try:
                output = task.run(task_input)
                
                if self.diagnostics_dir:
                    task.plot_diagnostics(output, self.diagnostics_dir / task_name)
                
                results[task_name] = TaskResult(output=output, success=True)
                
            except Exception as e:
                results[task_name] = TaskResult(
                    output=None,
                    success=False,
                    error=str(e)
                )
                break
        
        return results
