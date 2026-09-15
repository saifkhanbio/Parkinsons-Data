# ridge demographic models 2026 09 14

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [demographic_models.py](demographic_models.py): Controlled addition of age and sex to the original Hallmark ridge.
- [launch.py](launch.py): Run the fixed Step 2 nested comparison; all writes stay beside this script.
- [predict.py](predict.py): Apply a trusted local Step 2 model to aligned metadata and optional counts.
- [reporting.py](reporting.py): Paired fixed-prediction uncertainty and reports for Step 2.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
