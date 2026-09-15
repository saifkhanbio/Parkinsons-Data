# hallmark blood block ridge 2026 09 15

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [blood_models.py](blood_models.py): Hallmark RNA and measured-cell blocks with independently tuned L2 penalties.
- [launch.py](launch.py): Launch once, detached; status is checked only on user request.
- [predict.py](predict.py): Apply a trusted local Hallmark/blood model to aligned metadata and optional counts.
- [reporting.py](reporting.py): Paired repeated nested-CV uncertainty for Hallmark and measured-cell models.
- [worker.py](worker.py): Nested Hallmark plus measured-cell ridge comparison with separate penalties.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
