# svm learning curves 2026 09 12

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [launch.py](launch.py): Launch controlled nonlinear comparison and nested learning curves.
- [predict.py](predict.py): Apply a trusted SVM development artifact; scores are uncalibrated margins.
- [svm_analysis.py](svm_analysis.py): Fold-local SVM/ridge fitting and deterministic learning subsets.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
