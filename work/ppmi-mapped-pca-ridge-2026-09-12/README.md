# mapped pca ridge 2026 09 12

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [launch.py](launch.py): PCA plus ridge with a reproduced gene-space ridge benchmark.
- [pca_ridge.py](pca_ridge.py): Training-only PCA with a preserved gene-space ridge control.
- [predict.py](predict.py): Apply a trusted mapped-RNA artifact using its complete raw-count denominator.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
