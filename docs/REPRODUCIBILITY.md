# Environment and execution

## Environment snapshot

The accompanying requirements files record versions used in the local Python
environment and the separately installed boosting-model dependencies. The
analysis also used R 4.6.1, with DESeq2 1.52.0, BiocParallel 1.46.0 and jsonlite
2.0.0. Other R dependencies include edgeR, fgsea and msigdbr. Install compatible
Bioconductor/CRAN packages rather than copying the private workspace's installed
library tree. Gene-set release and mapping checks are defined in the R scripts.

Python scripts use NumPy, pandas, SciPy, scikit-learn and PyTorch, together with
the plotting, statistics and utility libraries in `requirements.txt`. Optional
boosting comparisons require `requirements-boosting.txt`. The Cytoscape network
scripts expect a locally running Cytoscape REST service.

Activate your environment before execution. Set `PPMI_PYTHON_ENV` to its directory
when using the QC shell launcher. Python launchers use `sys.executable` for their
child process. Some GPU validation routines explicitly select an RTX A3000 and
assert CUDA availability; they require review before use on other hardware.
DESeq2 fitting uses R/CPU, while compatible Python transformations and numerical
checks use PyTorch/CUDA. CPU-only execution is not supported by every script.

## Before running an analysis

1. Obtain authorized, release-matched inputs and configure the private paths in
   [DATA_ACCESS.md](DATA_ACCESS.md).
2. Resolve the early `outputs/` versus later `work/` layout. Restore required
   upstream artifacts before starting a dependent stage.
3. Read the stage source and README. Check input filenames, output destinations,
   environment settings and fixed study assertions.
4. Run `python tools/check_source.py` from the repository root.

R parsing and Python compilation do not execute the data analysis. Successful
syntax checks alone do not establish that the authorized inputs or numerical
environment have been reconstructed.

## Execution order

Follow the ordered stages in [WORKFLOW.md](WORKFLOW.md). For example, after
restoring private eligibility and expression-QC inputs and resolving paths:

```bash
python work/ppmi-model-design-2026-09-12/prepare_ppmi_covariates.py
Rscript work/ppmi-model-design-2026-09-12/check_ppmi_design.R
```

After reconstructing the classifier's approved inputs and upstream transforms:

```bash
python work/ppmi-mapped-gene-classifier-2026-09-12/launch.py
```

These are stage-specific examples, not a complete clean-checkout reproduction
command. Several launchers perform preflight checks and then start detached
workers. Review their process and thread settings before launch. Status files,
logs, trained estimators and predictions are private local outputs and should
remain outside version control.

## Statistical safeguards in the source

- Unique participant/sample alignment, finite non-negative integer counts,
  expression eligibility and full-rank design checks.
- Raw-count DESeq2 inference with explicit contrast, filtering and convergence
  handling, and collection-specific enrichment correction.
- Predictive transformations and supervised selection fitted within training
  partitions; matched splits for controlled comparisons.
- Fold-wise AUROC and average precision/PR summaries distinguished from pooled
  prediction summaries and threshold-dependent measures.
- Input hashing, solver checks, CPU/GPU comparisons where implemented, and
  saved-estimator inference checks.

Study-specific assertions are intentional. If a different data release fails a
check, investigate cohort, annotation, ordering and input provenance before
changing an expected value. The exploratory model-comparison modules represent
different study iterations; do not treat them as one jointly prespecified search.

## Reproduction changes

Private identifiers and machine-specific locations were parameterized for this
public copy, and embedded narrative reports were removed. See
[PUBLIC_COPY.md](PUBLIC_COPY.md). No fitted model, performance result or manuscript
text is bundled with this new source upload.
