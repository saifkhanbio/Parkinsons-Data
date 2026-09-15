# deseq2 2026 09 12

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [diagnose_influence.R](diagnose_influence.R)
- [fetch_gene_sets.R](fetch_gene_sets.R)
- [finish_analysis.py](finish_analysis.py): Process completed model artifacts while the DESeq2 runner continues.
- [gpu_postprocess.py](gpu_postprocess.py): CUDA result validation and sensitivity statistics; optionally await DESeq2.
- [gpu_validate.py](gpu_validate.py): Independent CUDA checks of counts, prefilter, and all five design matrices.
- [prepare_inputs.py](prepare_inputs.py): Reconstruct the all-sample sensitivity covariates without altering audits.
- [run_deseq2.R](run_deseq2.R)
- [run_enrichment.R](run_enrichment.R)
- [summarize_results.py](summarize_results.py): Validate saved results, compare sensitivity models, and produce a report.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
