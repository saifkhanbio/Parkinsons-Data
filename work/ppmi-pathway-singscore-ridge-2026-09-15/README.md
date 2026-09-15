# pathway singscore ridge 2026 09 15

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [export_sets.R](export_sets.R)
- [launch.py](launch.py): Launch once, detached; status is checked only on user request.
- [path_models.py](path_models.py): Training-specific length-normalized, non-directional singscore features.
- [predict.py](predict.py): Predict using a trusted saved score model and aligned full-panel raw counts.
- [prepare_mapping.py](prepare_mapping.py): Freeze outcome-independent annotation mapping and original feature lengths.
- [reference_scores.R](reference_scores.R)
- [reporting.py](reporting.py): Paired repeated nested-CV uncertainty for Hallmark and measured-cell models.
- [validation.py](validation.py): Independent R formula checks, training isolation, and inference tests.
- [worker.py](worker.py): Three annotation-only score classifiers on original repeated nested folds.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
