"""Minimal script to debug pipeline data flow."""

import numpy as np
from pathlib import Path
from btx.processing.core.pipeline import Pipeline, PipelineBuilder
from btx.processing.core.adapters import TaskAdapter, registry as task_registry
from btx.processing.btx_types import (
    LoadDataInput, LoadDataOutput,
    MakeHistogramInput, MakeHistogramOutput
)
from btx.processing.tasks import LoadData, MakeHistogram

def main():
    # Create minimal test data
    data = np.random.normal(10, 2, size=(100, 20, 20))  # 100 frames, 20x20 pixels
    I0 = np.ones(100)
    delays = np.linspace(0, 10, 100)
    on_mask = np.ones(100, dtype=bool)
    off_mask = ~on_mask

    # Minimal config
    config = {
        'setup': {
            'run': 123,
            'exp': 'debug_test',
        },
        'load_data': {
            'roi': [0, 20, 0, 20],
            'time_bin': 2.0,
        },
        'make_histogram': {
            'bin_boundaries': np.arange(5, 15, 0.5),
            'hist_start_bin': 1
        }
    }

    # Register tasks
    task_registry.register("load_data", LoadData)
    task_registry.register("make_histogram", MakeHistogram)

    # Create temporary directory for diagnostics
    diag_dir = Path("debug_diagnostics")
    diag_dir.mkdir(exist_ok=True)

    # Build minimal pipeline
    pipeline = (PipelineBuilder("Debug Pipeline")
        .add("load_data", task_registry.create("load_data", config))
        .add("make_histogram", task_registry.create("make_histogram", config),
             ["load_data"])
        .set_diagnostics_dir(diag_dir)
        .build())

    # Create input
    input_data = LoadDataInput(
        config=config,
        data=data,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=on_mask,
        laser_off_mask=off_mask
    )

    # Run pipeline
    results = pipeline.run(input_data)

    # Print results summary
    print("\n=== Results Summary ===")
    for task_name, result in results.results.items():
        print(f"\nTask: {task_name}")
        print(f"Success: {result.success}")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Output type: {type(result.output)}")
            if hasattr(result.output, '__dict__'):
                print("Output attributes:", list(result.output.__dict__.keys()))

if __name__ == "__main__":
    main()
