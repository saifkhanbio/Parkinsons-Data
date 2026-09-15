# Restricted inputs and private configuration

The public copy contains analysis source code. It does not redistribute PPMI
participant records, RNA-seq count files, blood-count measurements, sample
identifiers, fitted participant-level predictions or private decision lists.
Obtain the appropriate PPMI permissions and release-specific downloads directly
from the data provider.

## Source inputs

The original workflow uses Project 133 IR3/B38 phases 1–2 RNA-seq feature counts
and metadata; clinical visit, diagnosis, demographic, medication and collection
records; and the Blood Chemistry/Hematology table with its dictionary. The
analysis scripts retain their field names and provenance checks. Inputs must
refer to the matching release and participant/visit definitions.

Reference annotations and gene sets must also be obtained from their providers.
The enrichment and annotation scripts record or check the expected versions.
Gene-set and annotation databases are not bundled with this source release.

## Archive locations

Set the following variables to your authorized local files before running the
corresponding early stages:

```bash
export PPMI_IR3_ARCHIVE=/absolute/private/path/PPMI_RNAseq_IR3_Analysis.tar.gz
export PPMI_MEDICAL_ARCHIVE=/absolute/private/path/Medical.zip
export PPMI_PRIVATE_CONFIG=/absolute/private/path/private_config.json
```

These variables select files; they are not authentication credentials. Download
and authorize data access separately. The hematology audit expects its approved
archive at `work/ppmi-blood-counts-2026-09-12/incoming/Lab_Collection_Procedures.zip`.
That directory is ignored by version control.

## Private decision lists

Copy [private_config.example.json](../config/private_config.example.json) to a
private location and populate its lists from the approved local audit records:

- `medication_timing_exclusions`: identifiers selected by the medication timing audit.
- `excluded_control_ids`: identifiers excluded by the clinical eligibility audit.
- `qc_sensitivity_exclusions`: identifiers selected for the specific QC sensitivity refit.
- `influence_review_ids`: identifiers selected for the descriptive influence review.

All identifiers should be strings. Empty lists in the example describe the
configuration structure; they are not an analysis configuration. The original
study-specific checks remain and will reject incompatible inputs. Do not infer
the decision lists from classifier performance or change them to satisfy an
assertion. The QC validation step reads its omission list back from the locally
generated `preflight.json`.

## Early workspace layout

The eligibility script expects locally prepared `outputs/ppmi-extended-source/`
clinical CSV files and `outputs/ppmi-ir3-archive-check/` archive metadata and member
inventory. These restricted source-preparation artifacts are not included here.
The early eligibility, cohort, QC and design scripts write into `outputs/`,
whereas later scripts read the corresponding stages under `work/`.

Use a private staging layout that connects those locations consistently. The
original `outputs/ppmi-ir3-eligibility-audit` corresponds to
`work/ppmi-eligibility-audit/ir3-audit-2026-09-12`; other dated early stage directory
names are the same under `outputs/` and `work/`. Resolve these paths explicitly
before running a stage. Do not substitute normalized expression for raw counts
or reorder samples without reproducing the saved alignment checks.
