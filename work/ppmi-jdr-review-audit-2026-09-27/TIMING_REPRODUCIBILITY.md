# Timing-stratified model specification

## Cohort and count preparation

Use the frozen CBC-matched participant manifest and month-gap variable. Assign
zero month gap to `same_month`, and gaps of minus one or minus two to
`preceding_months`. The strata must be disjoint and together reproduce the
complete parent membership. Retrieve raw integer counts from the original
DESeq2 object in the participant order of each stratum; retain the upstream
gene universe. Do not reuse its size factors, dispersions or fitted coefficients.

Drop unused batch levels, recenter age, RIN and intergenic proportion within
stratum, and standardize the five measured-cell covariates within stratum.
Require full-rank model matrices before fitting. The reference specification is
`~ batch + age_c + sex + RIN_c + intergenic_c + group`. The adjusted specification
adds `log_wbc_z`, `neutrophils_z`, `monocytes_z`, `eosinophils_z` and
`basophils_z` before `group`. The disease contrast is PD versus Control.

## DESeq2 inference

Each model starts from a fresh count dataset. Use Wald inference, parametric
dispersion fitting, ratio size factors, `betaPrior=FALSE`, and
`minReplicatesForReplace=Inf`. Retry nonconverged coefficients with
`nbinomWaldTest(..., maxit=1000)`. Retain all gene identities in output, but
exclude persistent nonconvergence from inference and enrichment ranks.
Save convergence flags, dispersion estimates, Cook's diagnostics and size factors.

Apply DESeq2 independent filtering with `alpha=0.05`, and also save BH-adjusted
P values without independent filtering. The finalizer independently checks both
BH calculations on their appropriate testable sets. Fixed candidate comparisons
are descriptive; significance is taken from each model's genome-wide correction.

## Hallmark enrichment and eligibility

Use the same frozen Hallmark annotation as upstream analyses. Rank converged
genes with finite Wald statistics and P values, remove Ensembl version suffixes,
require unique mapped identifiers, and break ties by identifier.

Use `fgseaMultilevel` with `minSize=15`, `maxSize=500`, `eps=0`, `sampleSize=101`,
`nPermSimple=10000`, `scoreType="std"`, seed `20260912` and a serial backend.
Apply BH correction over the eligible tested collection. Save coverage for every
annotated Hallmark and verify that tested sets exactly match those with 15–500
ranked members. The collection size and the number of eligible tested sets need
not be equal after gene-level exclusions. Eligibility thresholds are fixed.

The public fitting script uses the eligibility-aware check also implemented in
`finalize_enrichment.R`, which can finish enrichment from preserved ranks without
refitting RNA models. No observed pathway counts or completed-study findings are
embedded in this documentation.

## Traceability

The launcher hashes the raw-count object, annotation, CBC covariates and source
files before starting, and checks these hashes after fitting. The finalizer
checks them again. Save the generated R session information with local outputs.
Do not change source files or input data during an active run, and do not mix
checkpoints across different inputs or settings.
