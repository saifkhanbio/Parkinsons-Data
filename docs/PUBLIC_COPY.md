# Public-copy preparation

This release was prepared from the local analysis workspace on 15 September
2026. Original analysis files and completed computations were preserved locally.
Only explicitly selected Python, R and shell source files were copied.
Documentation was written separately because the original READMEs also contain
participant identifiers and unpublished result summaries.

The public copy makes the following packaging changes:

- Participant identifiers embedded in cohort selection, QC sensitivity and
  influence review are read from private configuration or the local audit output.
- Two machine-specific archive paths are supplied through environment variables.
- Detached Python launches inherit the active interpreter environment. The QC
  shell launcher uses a relative project path and an explicit environment path.
- Six embedded narrative report templates are replaced with a short pointer to
  the locally generated tables and diagnostics. Numerical calculations and
  tabular-output code remain available.

These changes preserve statistical computations, model definitions, tuning
grids and training-only preprocessing. Source syntax and the file allowlist
were checked; no data-dependent analysis was rerun for this upload. Software
test fixtures and held-out perturbation checks remain source-code tests and are
separate from participant data used for the scientific analyses.

Private study branches, abandoned data-retrieval work, earlier scheduler copies,
installed dependency trees and manuscript-production files are outside this
release. The completed source directories retain their original names for
cross-stage traceability. The earlier solver implementation is retained alongside
the revised solver code so that the numerical change remains inspectable.

The pre-existing `Supplementary material.zip` is preserved byte-for-byte in the
GitHub tree. No new archive, figure, aggregate-result table or manuscript file
is added by this release.
