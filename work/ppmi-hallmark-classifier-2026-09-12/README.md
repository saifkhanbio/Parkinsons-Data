# hallmark classifier 2026 09 12

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [export_hallmark.R](export_hallmark.R)
- [launch.py](launch.py): Validate and detach the fixed 50-Hallmark classifier experiment.
- [pathways.py](pathways.py): Training-only, equal-weight Hallmark expression features and logistic tuning.
- [predict.py](predict.py): Apply a trusted Hallmark classifier artifact and its saved training transform.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
