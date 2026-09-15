# Parkinsons-Data

Analysis code for PPMI baseline whole-blood RNA sequencing, measured hematology,
pathway analysis and classification of Parkinson’s disease.

This code release contains Python, R and shell analysis scripts, together with
documentation for their inputs, dependencies and execution order. Raw data,
participant records, expression matrices, fitted models, predictions, new result
files, figures and manuscript documents are not included in this upload.

The previously uploaded [Supplementary material.zip](Supplementary%20material.zip)
is retained unchanged. That existing archive is separate from this code release.

## Start here

- [Analysis workflow and module index](docs/WORKFLOW.md)
- [Environment and execution](docs/REPRODUCIBILITY.md)
- [Restricted inputs and private configuration](docs/DATA_ACCESS.md)
- [Public-copy changes](docs/PUBLIC_COPY.md)

The source retains its dated `work/` directories so that relative imports and
cross-stage paths remain traceable. Each included analysis directory has a
code-only README. The scripts implement cohort eligibility and expression QC,
multivariable DESeq2 models, Hallmark/C2/GO enrichment, immune-marker analyses,
measured-cell adjustment, gene annotation and influence review, and controlled
classifier comparisons. Additional modules cover demographic assessments,
class weighting, stratified modeling, blood-cell covariates and calibration.

## Quick check

From the repository root, run:

```bash
python tools/check_source.py
```

This checks source syntax and the publication file allowlist without loading
participant data or running analyses. Analysis execution additionally requires
authorized PPMI inputs and the environment described in the documentation.

## Data access

Participant-level source data must be obtained directly through approved access
to the [Parkinson’s Progression Markers Initiative](https://www.ppmi-info.org/)
and its [data portal](https://ida.loni.usc.edu/). Access to this repository does
not confer access to PPMI data. Keep restricted inputs and locally generated
participant-level outputs outside version control.

## Scope of reproducibility

This is a source release of the study’s analysis scripts. It preserves
study-specific assertions, fold definitions, tuning grids and source checks.
It is not a general-purpose package or an automatically executable workflow for
arbitrary cohorts. Restore the required authorized inputs, review file paths and
use the documented stage order before launching a computation. No study models
were refitted as part of preparing this code release.

No new software license is assigned by this upload. PPMI data access and use
conditions remain separate from the availability of the source code.
