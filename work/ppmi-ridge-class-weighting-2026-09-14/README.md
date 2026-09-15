# ridge class weighting 2026 09 14

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [launch.py](launch.py): Execute the prespecified RNA ridge weighting comparison.
- [report.py](report.py): Paired discrimination and descriptive calibration/operating-point assessment.
- [weighting.py](weighting.py): Training-only weighting selection for the original RNA-only ridge.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
