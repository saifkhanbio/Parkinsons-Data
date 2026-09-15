# classifier refinement retry 2026 09 12

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [elastic_solver.py](elastic_solver.py): Convex elastic-net logistic regression using nonnegative coefficient splitting.
- [launch.py](launch.py): Validate and detach a reproducible classifier refinement job.
- [predict.py](predict.py): Apply a trusted refinement artifact with its saved training-selected threshold.
- [refine.py](refine.py): Nested elastic-net/ridge refinement with training-only operating thresholds.
- [validate_solver_repair.py](validate_solver_repair.py): Check repaired optimizer against SAGA and the failed real training partition.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
