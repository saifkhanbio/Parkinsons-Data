"""Audit supplied hematology records and conservative screening-to-RNA linkage."""

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'incoming/Lab_Collection_Procedures.zip'
COHORT = ROOT.parent / 'ppmi-expression-qc-2026-09-12/cohort_after_QC.json'
OUT = ROOT / 'audit'
# Source test names and unit strings were inspected before defining this mapping.
TESTS = {
    'HMT7': ('wbc', 'GI/L', 'x10^3/uL', 1),
    'HMT8': ('neutrophils_absolute', 'GI/L', 'x10^3/uL', 1),
    'HMT9': ('lymphocytes_absolute', 'GI/L', 'x10^3/uL', 1),
    'HMT10': ('monocytes_absolute', 'GI/L', 'x10^3/uL', 1),
    'HMT11': ('eosinophils_absolute', 'GI/L', 'x10^3/uL', 1),
    'HMT12': ('basophils_absolute', 'GI/L', 'x10^3/uL', 1),
    'HMT15': ('neutrophils_percent', '%', '%', 1),
    'HMT16': ('lymphocytes_percent', '%', '%', 1),
    'HMT17': ('monocytes_percent', '%', '%', 1),
    'HMT18': ('eosinophils_percent', '%', '%', 1),
    'HMT19': ('basophils_percent', '%', '%', 1),
    'HMT13': ('platelets', 'GI/L', 'x10^3/uL', 1),
    'HMT3': ('rbc', 'TI/L', 'x10^6/uL', 1),
    'HMT40': ('hemoglobin', 'g/L', 'g/dL', 0.1),
    'HMT2': ('hematocrit_fraction', '', '%', 100),
}
FRACTIONS = [spec[0] for spec in TESTS.values() if spec[0].endswith('_percent')]
CORE = ['wbc'] + FRACTIONS
PANEL_KEYS = ['PATNO', 'EVENT_ID', 'LCOLLDT', 'COLLTM', 'LABCODE', 'PAG_NAME', 'LVISTYPE']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def month_index(value):
    parsed = datetime.strptime(value, '%m/%Y')
    return parsed.year * 12 + parsed.month


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def write_tsv(name, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    original_hashes = {'archive': sha256(ARCHIVE), 'cohort': sha256(COHORT)}
    cohort_rows = json.loads(COHORT.read_text())
    cohort = {row['PATNO']: row for row in cohort_rows}
    assert len(cohort_rows) == len(cohort) == 558
    assert Counter(row['metadata_label'] for row in cohort_rows) == {'PD': 379, 'Control': 179}
    assert all(row['visit'] == 'BL' and len(row['collection_months']) == 1 for row in cohort_rows)
    panels_raw = defaultdict(list)
    test_inventory = Counter()
    date_issues = []
    latest_rna_months = defaultdict(set)
    archive_inventory = []
    total_lab_rows = 0
    cohort_hematology_rows = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None, 'Archive CRC failure'
        names = [name for name in archive.namelist() if 'Hematology' in name]
        assert len(names) == 1
        member = names[0]
        for info in archive.infolist():
            with archive.open(info) as handle:
                fields = next(csv.reader(io.TextIOWrapper(handle, encoding='utf-8-sig')))
            archive_inventory.append({'member': info.filename, 'bytes': info.file_size, 'columns': fields})
        # Keep an exact local copy of the requested table, without extracting other files.
        destination = ROOT / 'incoming' / Path(member).name
        digest = hashlib.sha256()
        with archive.open(member) as source, destination.open('wb') as target:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
                target.write(block)
        assert sha256(destination) == digest.hexdigest()
        with destination.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            source_fields = reader.fieldnames
            assert set(PANEL_KEYS + ['LTSTCODE', 'LTSTNAME', 'LSIRES', 'LSIUNIT', 'LUSRES', 'LUSUNIT']) <= set(source_fields)
            with gzip.open(OUT / 'cohort_hematology_source_rows.tsv.gz', 'wt', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=['source_member', 'source_csv_row'] + source_fields, delimiter='\t')
                writer.writeheader()
                for row_number, row in enumerate(reader, 2):
                    total_lab_rows += 1
                    if row['PATNO'] not in cohort or not row['LTSTCODE'].startswith('HMT'):
                        continue
                    cohort_hematology_rows += 1
                    row = {'source_member': member, 'source_csv_row': row_number, **row}
                    writer.writerow(row)
                    test_inventory[(row['LTSTCODE'], row['LTSTNAME'], row['LSIUNIT'], row['LUSUNIT'])] += 1
                    try:
                        month_index(row['LCOLLDT'])
                    except ValueError:
                        date_issues.append({'source_csv_row': row_number, 'PATNO': row['PATNO'], 'date': row['LCOLLDT']})
                        continue
                    panels_raw[tuple(row[key] for key in PANEL_KEYS)].append(row)
        research = [name for name in archive.namelist() if name.startswith('Research_Biospecimens_')]
        assert len(research) == 1
        with archive.open(research[0]) as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig')):
                if row['PATNO'] in cohort and row['EVENT_ID'] == 'BL' and row['BLDRNA'] == '1':
                    latest_rna_months[row['PATNO']].add(row['BLDDRDT'])
    assert len(latest_rna_months) == 558
    assert all(latest_rna_months[pid] == set(row['collection_months']) for pid, row in cohort.items())
    assert not date_issues, 'Unparseable cohort hematology dates require review'
    panels = []
    panel_index = defaultdict(list)
    result_issues = []
    provenance = []
    for index, (key, rows) in enumerate(sorted(panels_raw.items()), 1):
        panel = dict(zip(PANEL_KEYS, key))
        pid = panel['PATNO']
        panel.update(panel_id=f'panel_{index:05d}', group=cohort[pid]['metadata_label'],
                     rna_collection_month=cohort[pid]['collection_months'][0])
        panel['month_gap_lab_minus_rna'] = month_index(panel['LCOLLDT']) - month_index(panel['rna_collection_month'])
        by_code = defaultdict(list)
        for row in rows:
            by_code[row['LTSTCODE']].append(row)
        for code, (name, si_unit, us_unit, factor) in TESTS.items():
            candidates = by_code[code]
            evidence = {'panel_id': panel['panel_id'], 'PATNO': pid, 'test_code': code, 'variable': name,
                        'source_csv_rows': ';'.join(str(row['source_csv_row']) for row in candidates)}
            values = []
            reasons = []
            for row in candidates:
                si, us = numeric(row['LSIRES']), numeric(row['LUSRES'])
                reason = None
                if row['LSIUNIT'] != si_unit or row['LUSUNIT'] != us_unit:
                    reason = 'unexpected_unit'
                elif si is None:
                    reason = 'missing_or_nonnumeric_SI_result'
                elif si < 0 or (name.endswith('_percent') and si > 100) or (name == 'hematocrit_fraction' and si > 1):
                    reason = 'invalid_numeric_range'
                elif name in ['wbc', 'rbc', 'hemoglobin', 'hematocrit_fraction', 'platelets'] and si == 0:
                    reason = 'zero_total_count_or_red_cell_measure_review'
                elif us is None:
                    reason = 'missing_or_nonnumeric_US_result'
                else:
                    # Allow the rounding uncertainty in both displayed representations.
                    si_step = float(Decimal(row['LSIRES']).as_tuple().exponent)
                    us_step = float(Decimal(row['LUSRES']).as_tuple().exponent)
                    tolerance = (10 ** si_step * factor + 10 ** us_step) / 2 + 1e-9
                    if abs(si * factor - us) > tolerance:
                        reason = 'SI_US_conversion_disagreement'
                if reason:
                    reasons.append(reason)
                    result_issues.append({**evidence, 'source_csv_row': row['source_csv_row'], 'reason': reason,
                                          'LSIRES': row['LSIRES'], 'LUSRES': row['LUSRES'],
                                          'LSIUNIT': row['LSIUNIT'], 'LUSUNIT': row['LUSUNIT'],
                                          'LRESFLG': row['LRESFLG'], 'LTSTCOMM': row['LTSTCOMM']})
                else:
                    values.append(si)
            if not candidates:
                reasons.append('test_absent')
            if len(set(values)) > 1:
                reasons.append('conflicting_duplicate_results')
            panel[name] = values[0] if values and not reasons else None
            evidence['status'] = ';'.join(sorted(set(reasons))) if reasons else ('identical_duplicates' if len(values) > 1 else 'numeric_units_verified')
            provenance.append(evidence)
        panel['complete_wbc_and_five_percentages'] = all(panel[name] is not None for name in CORE)
        panel['complete_all_15_measures'] = all(panel[spec[0]] is not None for spec in TESTS.values())
        panel['five_percentages_sum'] = sum(panel[name] for name in FRACTIONS) if all(panel[name] is not None for name in FRACTIONS) else None
        panel['percentage_sum_review_flag'] = panel['five_percentages_sum'] is not None and abs(panel['five_percentages_sum'] - 100) > 0.3
        extra_percentages = []
        for code, name in [('HMT96', 'atypical_lymphocytes_percent'), ('HMT21', 'bands_percent'), ('HMT69', 'metamyelocytes_percent')]:
            source = by_code[code]
            values = [numeric(r['LSIRES']) for r in source if r['LSIUNIT'] == '%']
            panel[name] = values[0] if len(source) == len(values) == 1 and values[0] is not None else None
            if panel[name] is not None:
                extra_percentages.append(panel[name])
        panel['five_plus_reported_additional_percentages_sum'] = (panel['five_percentages_sum'] + sum(extra_percentages)) if panel['five_percentages_sum'] is not None else None
        differences = []
        for cell in ['neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils']:
            if all(panel[name] is not None for name in ['wbc', cell + '_percent', cell + '_absolute']):
                differences.append(abs(panel[cell + '_absolute'] - panel['wbc'] * panel[cell + '_percent'] / 100))
        panel['max_abs_differential_reconstruction_difference'] = max(differences) if differences else None
        # These diagnostics are review flags; they do not automatically exclude panels.
        panels.append(panel)
        panel_index[pid].append(panel)
    selections = []
    participant_coverage = []
    for row in cohort_rows:
        pid = row['PATNO']
        available = panel_index[pid]
        screening = [panel for panel in available if panel['EVENT_ID'] == 'SC' and panel['month_gap_lab_minus_rna'] <= 0]
        nearest_gap = max((panel['month_gap_lab_minus_rna'] for panel in screening), default=None)
        nearest = [panel for panel in screening if panel['month_gap_lab_minus_rna'] == nearest_gap]
        summary = {'PATNO': pid, 'group': row['metadata_label'], 'rna_collection_month': row['collection_months'][0],
                   'any_hematology': bool(available), 'baseline_visit_panels': sum(p['EVENT_ID'] == 'BL' for p in available),
                   'same_month_panels': sum(p['month_gap_lab_minus_rna'] == 0 for p in available),
                   'nearest_nonfuture_screening_month_gap': nearest_gap,
                   'nearest_screening_panel_count': len(nearest),
                   'selection_status': 'unique_screening_panel' if len(nearest) == 1 else ('ambiguous_screening_panels' if nearest else 'no_nonfuture_screening_panel'),
                   'selected_panel_id': nearest[0]['panel_id'] if len(nearest) == 1 else ''}
        participant_coverage.append(summary)
        selected = nearest[0] if len(nearest) == 1 else {}
        selections.append({**summary, **{k: v for k, v in selected.items() if k not in summary}})
    assert len(selections) == len({r['PATNO'] for r in selections}) == 558
    assert [r['PATNO'] for r in selections] == [r['PATNO'] for r in cohort_rows]
    assert all(r.get('EVENT_ID') == 'SC' and r['month_gap_lab_minus_rna'] <= 0 for r in selections if r['selected_panel_id'])
    coverage = []
    for group in ['PD', 'Control', 'All']:
        members = [r for r in selections if group == 'All' or r['group'] == group]
        for window, minimum in [('same_calendar_month', 0), ('same_or_previous_calendar_month', -1), ('same_or_previous_two_calendar_months', -2), ('any_nonfuture_screening_month', -100000)]:
            eligible = [r for r in members if r['selected_panel_id'] and minimum <= r['month_gap_lab_minus_rna'] <= 0]
            record = {'group': group, 'window': window, 'cohort_n': len(members), 'unique_panel_n': len(eligible),
                      'complete_wbc_and_five_percentages_n': sum(r['complete_wbc_and_five_percentages'] for r in eligible),
                      'complete_all_15_measures_n': sum(r['complete_all_15_measures'] for r in eligible)}
            for spec in TESTS.values():
                record[spec[0] + '_n'] = sum(r[spec[0]] is not None for r in eligible)
            coverage.append(record)
    write_tsv('participant_coverage.tsv', participant_coverage)
    write_tsv('screening_candidates.tsv', selections)
    write_tsv('all_cohort_panels.tsv', panels)
    write_tsv('panel_result_provenance.tsv', provenance)
    write_tsv('result_issues.tsv', result_issues)
    write_tsv('coverage_by_group_and_window.tsv', coverage)
    write_tsv('test_inventory.tsv', [dict(zip(['LTSTCODE', 'LTSTNAME', 'LSIUNIT', 'LUSUNIT'], key), cohort_rows=n) for key, n in sorted(test_inventory.items())])
    gap_counts = Counter((r['group'], r['nearest_nonfuture_screening_month_gap'], r['selection_status']) for r in participant_coverage)
    write_tsv('nearest_screening_month_gaps.tsv', [{'group': key[0], 'month_gap': key[1], 'status': key[2], 'participants': n} for key, n in sorted(gap_counts.items(), key=str)])
    assert original_hashes == {'archive': sha256(ARCHIVE), 'cohort': sha256(COHORT)}
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': original_hashes,
        'hematology_member': member, 'hematology_sha256': digest.hexdigest(),
        'archive_crc_pass': True, 'total_hematology_table_rows': total_lab_rows,
        'cohort_hematology_rows': cohort_hematology_rows, 'cohort_n': 558,
        'participants_with_any_hematology': sum(r['any_hematology'] for r in participant_coverage),
        'participants_with_baseline_visit_hematology': sum(r['baseline_visit_panels'] > 0 for r in participant_coverage),
        'participants_with_same_month_hematology': sum(r['same_month_panels'] > 0 for r in participant_coverage),
        'selection_status_counts': dict(Counter(r['selection_status'] for r in participant_coverage)),
        'result_issue_counts_all_visits': dict(Counter(r['reason'] for r in result_issues)),
        'screening_percentage_sum_review_flags': sum(r.get('percentage_sum_review_flag', False) for r in selections),
        'screening_sum_flags_explained_by_reported_extra_categories': sum(r.get('percentage_sum_review_flag', False) and r.get('five_plus_reported_additional_percentages_sum') is not None and abs(r['five_plus_reported_additional_percentages_sum'] - 100) <= 0.3 for r in selections),
        'rna_months_reconfirmed_from_new_research_biospecimens': 558,
        'coverage': coverage, 'date_precision': 'calendar_month_only',
        'panel_identity_limit': 'Collection time is not a day-level identifier; panel keys are operational groupings.',
        'selection_rule': 'Nearest screening month at/before RNA month; require one panel in that month; no selection based on result completeness; no fallback for missing results or ties.',
        'interpretation': 'Candidate screening covariates only. Same-month screening is not same-day or baseline-visit evidence. No blood-cell-adjusted model was fitted.',
    }
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'archive_inventory.json').write_text(json.dumps(archive_inventory, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'coverage'}, indent=2))
    for row in coverage:
        print({key: row[key] for key in ['group', 'window', 'unique_panel_n', 'complete_wbc_and_five_percentages_n', 'complete_all_15_measures_n']})


if __name__ == '__main__':
    main()
