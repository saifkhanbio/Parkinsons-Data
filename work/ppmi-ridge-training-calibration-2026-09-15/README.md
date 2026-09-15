# ridge training calibration 2026 09 15

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [apply_calibrator.py](apply_calibrator.py): Apply one saved fold calibration map to that model's original probabilities.
- [calibrator.py](calibrator.py): Positive-slope logistic calibration; no raw RNA or held-out labels accepted.
- [launch.py](launch.py): Launch once, detached; status is checked only on user request.
- [worker.py](worker.py)

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
