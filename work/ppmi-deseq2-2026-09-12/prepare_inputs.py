"""Reconstruct the all-sample sensitivity covariates without altering audits."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
QC = WORK / "ppmi-expression-qc-2026-09-12"
DESIGN = WORK / "ppmi-model-design-2026-09-12"
MANIFEST = WORK / "ppmi-broader-cohort-2026-09-12/cohort_manifest.json"
ARCHIVE = WORK / "ppmi-eligibility-audit/incoming/download.zip"


def month_number(value):
    assert isinstance(value, str) and re.fullmatch(r"\d{2}/\d{4}", value)
    month, year = map(int, value.split("/"))
    assert 1 <= month <= 12
    return year * 12 + month


def main():
    manifest = json.loads(MANIFEST.read_text())
    metadata = pd.read_csv(QC / "sample_metadata.tsv", sep="\t", dtype=str)
    assert metadata.PATNO.is_unique
    metadata = metadata.set_index("PATNO")
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open("Demographics_11Sep2026.csv") as handle:
            demographics = pd.read_csv(handle, dtype=str)
    rows, evidence = [], []
    for record in manifest:
        participant = record["PATNO"]
        source = demographics[demographics.PATNO == participant]
        births = source.BIRTHDT.dropna().unique()
        sexes = source.SEX.dropna().unique()
        assert len(births) == len(sexes) == 1, participant
        assert len(record["collection_months"]) == 1, participant
        sex = {"0": "Female", "1": "Male"}[sexes[0]]
        meta = metadata.loc[participant]
        assert sex == meta.GENDER and record["sample"] == meta.Sample
        assert record["analysis_group"] == meta.group
        age = (month_number(record["collection_months"][0]) - month_number(births[0])) / 12
        assert 18 <= age <= 100
        phase = meta.Sample.split(".")[0].replace("-IR1", "")
        rows.append(dict(
            PATNO=participant, group=record["analysis_group"],
            age_collection_years=age, sex=sex, phase=phase, plate=meta.Plate,
            batch=phase + "__plate_" + meta.Plate,
            RIN=float(meta["RIN Value"]),
            intergenic_percent=float(meta.PCT_INTERGENIC_BASES),
            usable_percent=float(meta.PCT_USABLE_BASES),
            medication_timing_sensitivity_exclude=record["exclude_in_medication_timing_sensitivity"],
        ))
        evidence.append(dict(PATNO=participant,
                             demographics_csv_rows=[int(i) + 2 for i in source.index],
                             age_method="released collection minus birth year-month / 12",
                             archive_member="Demographics_11Sep2026.csv"))
    all_data = pd.DataFrame(rows)
    assert len(all_data) == 579 and all_data.PATNO.is_unique
    assert all_data.group.value_counts().to_dict() == {"PD": 393, "Control": 186}
    assert not all_data.isna().any().any()
    primary = pd.read_csv(DESIGN / "DESeq2_colData.tsv", sep="\t", dtype={"PATNO": str})
    assert len(primary) == 558 and primary.PATNO.is_unique
    aligned = all_data.set_index("PATNO").loc[primary.PATNO]
    for column in ["group", "sex", "phase", "batch"]:
        assert aligned[column].tolist() == primary[column].tolist(), column
    for column in ["age_collection_years", "RIN", "intergenic_percent", "usable_percent"]:
        np.testing.assert_allclose(aligned[column], primary[column], atol=1e-10)
    all_data.to_csv(OUT / "all_sample_covariates.tsv", sep="\t", index=False)
    (OUT / "all_sample_covariate_provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    inputs = [MANIFEST, ARCHIVE, QC / "raw_counts.tsv.gz", QC / "sample_metadata.tsv",
              QC / "cohort_after_QC.json", DESIGN / "DESeq2_colData.tsv",
              DESIGN / "DESeq2_design.rds", OUT / "ANALYSIS_PLAN.md"]
    hashes = {}
    for path in inputs:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path.relative_to(WORK.parent))] = digest.hexdigest()
    (OUT / "input_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Validated all 579 covariates; reconstructed values agree with saved 558-sample design.")


if __name__ == "__main__":
    main()
