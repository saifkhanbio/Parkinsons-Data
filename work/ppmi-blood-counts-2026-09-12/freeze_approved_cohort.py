"""Freeze the user-approved 528-person screening blood-count cohort."""

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'approved_cohort_528'
CORE = ['wbc', 'neutrophils_percent', 'lymphocytes_percent', 'monocytes_percent',
        'eosinophils_percent', 'basophils_percent']


def read_tsv(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        return reader.fieldnames, list(reader)


def table_text(fields, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main():
    candidates = ROOT / 'audit/screening_candidates.tsv'
    provenance_path = ROOT / 'audit/panel_result_provenance.tsv'
    audit_path = ROOT / 'audit/summary.json'
    validation_path = ROOT / 'audit/independent_validation.json'
    parent_path = ROOT.parent / 'ppmi-expression-qc-2026-09-12/cohort_after_QC.json'
    inputs = [candidates, provenance_path, audit_path, validation_path, parent_path]
    hashes = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    fields, rows = read_tsv(candidates)
    parent = json.loads(parent_path.read_text())
    assert len(rows) == len(parent) == 558
    assert len({r['PATNO'] for r in rows}) == 558
    assert [r['PATNO'] for r in rows] == [r['PATNO'] for r in parent]
    assert all(a['group'] == b['metadata_label'] for a, b in zip(rows, parent))
    assert json.loads(validation_path.read_text())['independent_source_reconstruction'] == 'PASS'
    included, dispositions = [], []
    for row in rows:
        reasons = []
        if row['selection_status'] != 'unique_screening_panel':
            reasons.append(row['selection_status'])
        else:
            if int(row['month_gap_lab_minus_rna']) not in [-2, -1, 0]:
                reasons.append('outside_approved_calendar_month_window')
            if row['complete_wbc_and_five_percentages'] != 'True':
                reasons.append('incomplete_core_blood_counts')
        if not reasons:
            assert row['EVENT_ID'] == 'SC'
            assert row['panel_id'] == row['selected_panel_id']
            assert math.isfinite(float(row['wbc'])) and float(row['wbc']) > 0
            assert all(math.isfinite(float(row[name])) and 0 <= float(row[name]) <= 100 for name in CORE[1:])
            included.append(row)
        dispositions.append({'PATNO': row['PATNO'], 'group': row['group'],
                             'status': 'included' if not reasons else 'excluded_from_blood_count_subset',
                             'reasons': ';'.join(reasons), 'selected_panel_id': row['selected_panel_id'],
                             'month_gap_lab_minus_rna': row['month_gap_lab_minus_rna']})
    assert len(included) == 528
    assert Counter(r['group'] for r in included) == {'PD': 358, 'Control': 170}
    assert sum(d['status'] != 'included' for d in dispositions) == 30
    panel_ids = {r['selected_panel_id'] for r in included}
    assert len(panel_ids) == 528
    provenance_fields, provenance = read_tsv(provenance_path)
    provenance = [r for r in provenance if r['panel_id'] in panel_ids]
    assert len(provenance) == 528 * 15
    for panel_id in panel_ids:
        assert {r['variable'] for r in provenance if r['panel_id'] == panel_id} >= set(CORE)
    excluded = [d for d in dispositions if d['status'] != 'included']
    audit = json.loads(audit_path.read_text())
    summary = {
        'status': 'user_approved_frozen_cohort', 'approval_date': '2026-09-12',
        'user_instruction': 'Accept Same or preceding two months 358 170 528',
        'scope': 'Measured blood-cell covariate analysis subset; original 558-person primary cohort preserved.',
        'window': 'RNA collection calendar month or either of the two preceding calendar months',
        'allowed_lab_minus_rna_month_gaps': [-2, -1, 0],
        'selection': 'Existing unique nearest nonfuture screening panel, with complete WBC and five differential percentages; no fallback, averaging, or imputation.',
        'participants': 528, 'PD': 358, 'Control': 170, 'excluded_from_parent_subset': 30,
        'excluded_by_group': dict(Counter(r['group'] for r in excluded)),
        'exclusion_reason_combinations': dict(Counter(r['reasons'] for r in excluded)),
        'included_by_calendar_month_gap': dict(Counter(r['month_gap_lab_minus_rna'] for r in included)),
        'complete_all_15_measures': sum(r['complete_all_15_measures'] == 'True' for r in included),
        'retained_extra_cell_category_flags': sum(r['percentage_sum_review_flag'] == 'True' for r in included),
        'input_sha256': hashes, 'archive_sha256': audit['input_sha256']['archive'],
        'source_hematology_sha256': audit['hematology_sha256'],
        'validation': 'Unique participants and panels; exact parent order and group labels; finite core values; expected counts; per-test source evidence; prior independent source reconstruction passed.',
        'limitations': ['Calendar-month matching is not exact-day matching or a 60-day interval.',
                        'Screening blood counts are proxies for composition at baseline RNA collection.',
                        'Nine participants lack at least one additional non-core measure; they remain included under the approved rule.',
                        'Four panels have separately reported atypical lymphocytes; no renormalization or exclusion was applied.'],
        'expression_models_fitted': False,
    }
    outputs = {
        'cohort_manifest.tsv': table_text(fields, included),
        'participant_dispositions.tsv': table_text(list(dispositions[0]), dispositions),
        'panel_result_provenance.tsv': table_text(provenance_fields, provenance),
        'participant_ids.txt': ''.join(r['PATNO'] + '\n' for r in included),
        'summary.json': json.dumps(summary, indent=2) + '\n',
    }
    # Refuse to silently replace a frozen artifact with changed content.
    OUT.mkdir(exist_ok=True)
    for name, content in outputs.items():
        target = OUT / name
        if target.exists():
            assert target.read_bytes() == content.encode(), f'Frozen output differs: {name}'
    for name, content in outputs.items():
        target = OUT / name
        if not target.exists():
            target.write_text(content)
    assert hashes == {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    print(json.dumps({k: v for k, v in summary.items() if k not in ['input_sha256', 'limitations']}, indent=2))


if __name__ == '__main__':
    main()
