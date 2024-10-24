from typing import Dict, Any, Optional, Set, List, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Type
from dataclasses import dataclass
from pathlib import Path
from .task import Task

class PipelineBuilder:
    """Builder pattern for Pipeline construction."""
    
    def __init__(self, name: str):
        self.pipeline = Pipeline(name)
    
    def add(self, name: str, task: Task, depends_on: Optional[List[str]] = None) -> 'PipelineBuilder':
        """Add a task to the pipeline."""
        self.pipeline.add_task(name, task, depends_on)
        return self
    
    def set_diagnostics_dir(self, path: Path) -> 'PipelineBuilder':
        """Set directory for diagnostic outputs."""
        self.pipeline.diagnostics_dir = path
        return self
    
    def build(self) -> 'Pipeline':
        """Return the constructed pipeline."""
        return self.pipeline

@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    success: bool
    results: Dict[str, 'TaskResult']
    execution_order: Sequence[str]
    error: Optional[str] = None

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
    
    def run(self, initial_input: Optional[Any] = None) -> PipelineResult:
        """Run the pipeline, assuming valid DAG structure."""
        results = {}
        success = True
        error = None
        
        print("\n=== Starting Pipeline Execution ===")
        
        for task_name in self.execution_order:
            print(f"\n--- Executing task: {task_name} ---")
            task = self.tasks[task_name]
            deps = self.dependencies[task_name]
            
            try:
                # Determine input based on dependencies
                if not deps:
                    task_input = initial_input
                    print(f"Using initial input type: {type(task_input)}")
                elif len(deps) == 1:
                    # Single dependency
                    dep = next(iter(deps))
                    prev_output = results[dep].output
                    print(f"Previous output type from {dep}: {type(prev_output)}")
                    
                    # Wrap output in appropriate input type
                    if task_name == "make_histogram":
                        print("Creating MakeHistogramInput")
                        task_input = MakeHistogramInput(
                            config=task.config,
                            load_data_output=prev_output
                        )
                    else:
                        task_input = prev_output
                    print(f"Created input type: {type(task_input)}")
                else:
                    # Multiple dependencies
                    task_input = {dep: results[dep].output for dep in deps}
                
                # Run task
                print(f"Running task {task_name}")
                output = task.run(task_input)
                print(f"Task {task_name} completed with output type: {type(output)}")
                
                if self.diagnostics_dir:
                    task.plot_diagnostics(output, self.diagnostics_dir / task_name)
                
                results[task_name] = TaskResult(output=output, success=True)
                
            except Exception as e:
                print(f"ERROR in task {task_name}: {str(e)}")
                results[task_name] = TaskResult(
                    output=None,
                    success=False,
                    error=str(e)
                )
                success = False
                error = str(e)
                break
        
        return PipelineResult(
            success=success,
            results=results,
            execution_order=self.execution_order,
            error=error
        )
