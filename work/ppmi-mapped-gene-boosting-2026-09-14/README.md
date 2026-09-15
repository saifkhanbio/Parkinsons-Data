# mapped gene boosting 2026 09 14

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [estimators.py](estimators.py): Fixed alternative estimators on the unchanged mapped-gene representation.
- [launch.py](launch.py): Validate and launch additional mapped-gene estimators.
- [predict.py](predict.py): Apply a trusted local model to a TSV: sample ID, then complete raw Geneid columns.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
