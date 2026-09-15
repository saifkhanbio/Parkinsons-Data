# mapped top100 ridge 2026 09 12

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [launch.py](launch.py): Fixed top-100 mapped-gene ridge classifier.
- [predict.py](predict.py): Apply a trusted mapped-RNA artifact using its complete raw-count denominator.
- [top100.py](top100.py): Fixed top-100 ANOVA ridge with a separate all-genes control.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
