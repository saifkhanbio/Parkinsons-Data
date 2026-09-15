# sex age stratified de ridge 2026 09 14

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [de_workers.py](de_workers.py): Audited training-only R exports and restartable serial DE workers.
- [launch.py](launch.py): Restartable four-worker DE scheduler followed by matched holdout evaluation.
- [model_utils.py](model_utils.py): Benchmark-equivalent ridge with a training-selected gene-level FDR mask.
- [prepare_plan.py](prepare_plan.py): Freeze combined-stratum holdouts and matched demographic control plans.
- [reporting.py](reporting.py): Matched held-out comparisons, conditional intervals, and figures.
- [run_selector.R](run_selector.R)
- [weighted_metrics.py](weighted_metrics.py): Tie-aware AUROC and average precision with bootstrap multiplicities.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
