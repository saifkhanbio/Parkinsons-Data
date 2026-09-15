# sex pooled fdr only ridge 2026 09 14

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [launch.py](launch.py): Launch once, detached; status is checked only on user request.
- [model_utils.py](model_utils.py): Benchmark-equivalent ridge with a training-selected gene-level FDR mask.
- [plan_checks.py](plan_checks.py)
- [reporting.py](reporting.py): Matched FDR-only versus strict-screen and Hallmark ridge assessment.
- [weighted_metrics.py](weighted_metrics.py): Tie-aware AUROC and average precision with bootstrap multiplicities.
- [worker.py](worker.py): FDR-only masks from verified cached training DE, matched ridge assessment.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
