# Differential-expression analysis specification

This public methods note accompanies the source and is used in its provenance
hashing. It replaces the private workspace's result-linked documentation.

Use approved raw integer counts, frozen cohort definitions and the locally
recorded covariate design. Estimate the PD-versus-Control contrast with Control
as reference. Use DESeq2 median-ratio size factors, a parametric dispersion
trend, two-sided Wald tests, BH correction and independent filtering. Disable
automatic count replacement; keep nonconverged and untestable genes explicit.
The expression prefilter and expected design sizes remain in the source.

The primary formula is `~ batch + age_c + sex + RIN_c + intergenic_c + group`.
Planned alternative specifications address medication timing, phase-level
technical adjustment, usable-base adjustment and the original QC eligibility
universe. Refit normalization and dispersion for each model and verify design
rank, participant order and input provenance.

Use finite converged Wald statistics for ranked enrichment. Preserve versioned
gene-set provenance, mapping rules, measured set-size bounds and fixed random
seeds. Adjust within each complete collection/model. Compare directions and
effect estimates across specifications without selecting models by the number
of discoveries. The source retains the exact implementation settings.
