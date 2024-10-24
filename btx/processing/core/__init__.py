# btx/processing/core/__init__.py
"""Core pipeline framework components."""

# Import and expose all core components
from .task import Task
from .pipeline import Pipeline, PipelineResult, TaskResult
from .builders import PipelineBuilder
from .adapters import (
    TaskAdapter, TaskRegistry, register_task,
    registry as task_registry
)

__all__ = [
    'Task',
    'Pipeline',
    'PipelineResult',
    'TaskResult',
    'PipelineBuilder',
    'TaskAdapter',
    'TaskRegistry',
    'register_task',
    'task_registry'
]
