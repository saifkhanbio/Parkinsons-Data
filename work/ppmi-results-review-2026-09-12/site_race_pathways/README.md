# site_race_pathways

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../../docs/WORKFLOW.md),
[execution guide](../../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../../docs/DATA_ACCESS.md).

## Source files

- [finalize_gpu.py](finalize_gpu.py): Validate enrichment on CUDA and compare pathways with prior results.
- [run_enrichment.R](run_enrichment.R)

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
