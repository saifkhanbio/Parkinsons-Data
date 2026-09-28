# Additional sensitivity and evidence audits

Source code for batch-confounding diagnostics, site/race sensitivity,
elastic-net stability verification, and disjoint blood-count timing analyses.
This module extends the existing workflow; it does not replace upstream analyses.
Only code and methodological documentation are distributed here.

## Analyses

- Describe diagnosis composition within phase/plate batches and calculate the
  diagnosis-on-batch R-squared and variance inflation factor.
- On the saved outer folds, score participants using Laplace-smoothed diagnosis
  prevalence from their training batch. For unseen batches, use overall training
  prevalence. This is a batch-only diagnostic; a constant prediction for unseen
  batches does not establish that expression is free of technical confounding.
- Compare primary and site/race-adjusted effect estimates, standard errors,
  discovery transitions and Benjamini–Hochberg correction with independent
  filtering disabled, using preserved model outputs.
- Identify genes classified as stable across every protocol/family setting in
  the saved elastic-net analysis, and verify their per-repeat retention and
  sign consistency. Candidate names are read from local output rather than
  embedded in this public source.
- Refit reference and measured-cell-adjusted DESeq2 models in disjoint
  same-calendar-month and preceding-one-or-two-calendar-month CBC strata.
  Repeat normalization, dispersion estimation and coefficient fitting from raw
  integer counts; retain the upstream gene universe and fixed model settings.
- Recalculate Hallmark enrichment, eligible-set coverage and descriptive
  consistency of previously identified candidates and pathways.

The timing groups share a parent cohort and differ in participant composition.
These exploratory comparisons do not isolate a causal collection-timing effect;
same calendar month does not establish that measurements share a blood draw.

## Required private upstream files

All paths below are relative to `work/`. Generate or restore them through
authorized local access before executing the corresponding stage.

| Upstream module | Required files |
| --- | --- |
| `ppmi-deseq2-2026-09-12` | `primary_colData.tsv`, `primary/dds.rds`, `primary/results.tsv`, `primary/hallmark_enrichment.tsv`, `msigdb_hallmark_gobp.rds` |
| `ppmi-classifier-2026-09-12` | `metadata.tsv` |
| `ppmi-mapped-gene-classifier-2026-09-12` | `fold_plan.json` |
| `ppmi-results-review-2026-09-12` | `site_race_results.tsv` |
| `ppmi-classifier-feature-stability-2026-09-12` | `annotated_stability.tsv` |
| `ppmi-blood-cell-adjustment-2026-09-12` | `blood_covariates.tsv`, `cell_adjusted/results.tsv`, `robust_15_hallmark_comparison.tsv` |
| `ppmi-five-gene-review-2026-09-12` | `five_gene_review.tsv` |

Read those stages' documentation and the repository's
[execution guide](../../docs/REPRODUCIBILITY.md). The saved fold indices must
refer to the exact classifier metadata row order. Study-specific cohort totals,
gene counts and design checks are retained intentionally.

## Execution

Run from the repository root with the Python environment active and `Rscript`
on `PATH`:

```bash
python work/ppmi-jdr-review-audit-2026-09-27/audit.py
python work/ppmi-jdr-review-audit-2026-09-27/run_timing.py
python work/ppmi-jdr-review-audit-2026-09-27/finalize_timing.py
```

`audit.py` reads existing outputs; it does not refit RNA models. `run_timing.py`
first prepares both designs, then runs the two strata concurrently with four
DESeq2 CPU workers each and one BLAS thread per process. It also checks pathway
coverage and freezes input/source hashes. DESeq2 and fgsea run on CPU.
`finalize_timing.py` validates the finished files and writes numeric summaries.

To inspect the designs without fitting:

```bash
Rscript work/ppmi-jdr-review-audit-2026-09-27/timing_models.R same_month --prepare-only
Rscript work/ppmi-jdr-review-audit-2026-09-27/timing_models.R preceding_months --prepare-only
```

`finalize_enrichment.R` can regenerate missing enrichment from saved ranks
without refitting DESeq2. Supply one or both stratum names as arguments. Existing
enrichment files are validated against eligible-set coverage. Checkpoints belong
to their original inputs: use a fresh analysis directory if inputs or settings
change; do not reuse old fitted files with new input hashes.

## Outputs and validation

Execution writes local batch, sensitivity and stability tables; per-stratum
designs and metadata; model objects; gene statistics; convergence diagnostics;
enrichment ranks and coverage; input hashes; logs; and numeric summaries. These
are generated files, not part of the public source update. Keep participant data,
models, figures and result tables out of commits.

Checks cover participant uniqueness and alignment, finite nonnegative integer
counts, fixed gene identities, full design rank, convergence exclusions,
model-specific BH correction, eligible pathway membership, disjoint complete
cohort coverage and unchanged source inputs. See
[TIMING_REPRODUCIBILITY.md](TIMING_REPRODUCIBILITY.md) for fitting details.
