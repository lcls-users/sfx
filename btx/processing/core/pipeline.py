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
        """Add a task to the pipeline, maintaining addition order."""
        if name in self.tasks:
            raise ValueError(f"Task {name} already exists")
        
        self.tasks[name] = task
        self.dependencies[name] = set(depends_on or [])
        self.execution_order.append(name)
    
    def run(self, initial_input: Optional[Any] = None) -> Dict[str, TaskResult]:
        """Run the pipeline, assuming valid DAG structure."""
        results = {}
        
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            deps = self.dependencies[task_name]
            
            # Simplified input handling
            task_input = initial_input if not deps else \
                        results[next(iter(deps))].output if len(deps) == 1 else \
                        {dep: results[dep].output for dep in deps}
            
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
