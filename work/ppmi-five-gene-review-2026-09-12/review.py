"""Reference annotation, effects, counts, and conditional influence for five genes."""
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import subprocess
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
ANNOTATION = WORK / 'ppmi-gene-annotation-review-2026-09-12'
GTF = ANNOTATION / 'cache/gencode.v29.primary_assembly.annotation.gtf.gz'
SOURCES = {528: WORK / 'ppmi-blood-cell-adjustment-2026-09-12',
           445: WORK / 'ppmi-blood-cell-timing-2026-09-12'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def union_length(intervals):
    total, end = 0, -1
    for lo, hi in sorted(set(intervals)):
        total += max(0, hi - max(end, lo - 1))
        end = max(end, hi)
    return total


def main():
    assert not (ROOT / 'summary.json').exists(), 'Completed review exists; use a new directory for changed inputs.'
    models = ['subset_reference', 'cell_adjusted']
    primary = pd.read_csv(SOURCES[528] / 'cell_adjusted/results.tsv', sep='\t')
    genes = primary.loc[primary.padj.lt(.05), 'Geneid'].tolist()
    assert len(genes) == len(set(genes)) == 5
    inputs = [GTF, ANNOTATION / 'gencode_v29_gene_reference.tsv',
              WORK / 'ppmi-expression-qc-2026-09-12/gene_annotation.tsv',
              ROOT / 'review.py', ROOT / 'review_influence.R']
    for directory in SOURCES.values():
        inputs.append(directory / 'cell_adjusted/dds.rds')
        inputs.extend(directory / model / 'results.tsv' for model in models)
    hashes = {str(p): sha(p) for p in inputs}
    save_json('input_sha256.json', hashes)
    annotations, intervals, evidence = {}, defaultdict(lambda: defaultdict(list)), []
    attr = re.compile(r'(\w+) "([^"]*)"')
    gid = re.compile(r'gene_id "([^"]+)"')
    with gzip.open(GTF, 'rt') as f:
        for row, line in enumerate(f, 1):
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            assert len(fields) == 9
            if fields[2] not in ['gene', 'exon']:
                continue
            match = gid.search(fields[8])
            if match is None or match.group(1) not in genes:
                continue
            gene = match.group(1)
            if fields[2] == 'gene':
                assert gene not in annotations
                a = dict(attr.findall(fields[8]))
                annotations[gene] = {'Geneid': gene, 'reference_gene_name': a['gene_name'],
                                     'reference_gene_type': a['gene_type'], 'chromosome': fields[0],
                                     'start': int(fields[3]), 'end': int(fields[4]), 'strand': fields[6],
                                     'GTF_gene_line': row, 'match': 'exact_identifier_and_version'}
                evidence.append({'GTF_line': row, 'text': line.rstrip()})
            else:
                intervals[gene][(fields[0], fields[6])].append((int(fields[3]), int(fields[4])))
    assert set(annotations) == set(genes)
    counts_annotation = pd.read_csv(inputs[2], sep='\t').set_index('Geneid')
    old_reference = pd.read_csv(inputs[1], sep='\t').set_index('reference_Geneid')
    for gene in genes:
        length = sum(union_length(v) for v in intervals[gene].values())
        annotations[gene]['exon_union_length'] = length
        annotations[gene]['counted_length'] = int(counts_annotation.loc[gene, 'Length'])
        assert length == annotations[gene]['counted_length']
        assert annotations[gene]['reference_gene_name'] == old_reference.loc[gene, 'reference_gene_name']
    ann = pd.DataFrame([annotations[gene] for gene in genes])
    ann.to_csv(ROOT / 'annotations.tsv', sep='\t', index=False)
    save_json('GTF_source_evidence.json', evidence)
    effects = []
    for window, directory in SOURCES.items():
        for model in models:
            r = pd.read_csv(directory / model / 'results.tsv', sep='\t').set_index('Geneid').loc[genes].reset_index()
            assert r.beta_converged.all()
            r = r[['Geneid', 'log2FoldChange', 'lfcSE', 'pvalue', 'padj', 'max_cooks', 'cook_review_flag']]
            r['window'] = window; r['model'] = model
            r['fold_change_PD_Control'] = 2 ** r.log2FoldChange
            r['FC_CI95_lower'] = 2 ** (r.log2FoldChange - 1.959963984540054 * r.lfcSE)
            r['FC_CI95_upper'] = 2 ** (r.log2FoldChange + 1.959963984540054 * r.lfcSE)
            effects.append(r)
    effects = pd.concat(effects, ignore_index=True).merge(ann, on='Geneid', validate='many_to_one')
    assert len(effects) == 20
    effects.to_csv(ROOT / 'effect_estimates_all_four_models.tsv', sep='\t', index=False)
    save_json('status.json', {'status': 'RUNNING_TARGETED_INFLUENCE', 'candidates': genes})
    with (ROOT / 'influence.log').open('w') as log:
        subprocess.run(['/usr/local/bin/Rscript', str(ROOT / 'review_influence.R')], cwd=WORK.parent,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    obs = pd.read_csv(ROOT / 'sample_counts_and_influence.tsv', sep='\t', dtype={'PATNO': str})
    loo = pd.read_csv(ROOT / 'conditional_omission_checks.tsv', sep='\t', dtype={'PATNO_omitted': str})
    assert len(obs) == (528 + 445) * 5
    assert not obs.duplicated(['window', 'Geneid', 'PATNO']).any()
    assert not loo.duplicated(['window', 'Geneid', 'PATNO_omitted']).any()
    assert set(loo.Geneid) == set(genes) and set(loo.window) == {528, 445}
    assert loo.converged.all(), 'An omission did not converge; review before completion'
    statistics = []
    for (window, gene, group), rows in obs.groupby(['window', 'Geneid', 'group']):
        statistics.append({'window': int(window), 'Geneid': gene, 'group': group, 'n': len(rows),
                           'raw_min': int(rows.raw_count.min()), 'raw_median': float(rows.raw_count.median()),
                           'raw_max': int(rows.raw_count.max()), 'zero_count_n': int(rows.raw_count.eq(0).sum()),
                           'counts_ge_10_n': int(rows.raw_count.ge(10).sum()),
                           'normalized_median': float(rows.normalized_count.median()),
                           'normalized_q25': float(rows.normalized_count.quantile(.25)),
                           'normalized_q75': float(rows.normalized_count.quantile(.75)),
                           'largest_observation_share_raw_sum': float(rows.raw_count.max() / rows.raw_count.sum())})
    pd.DataFrame(statistics).to_csv(ROOT / 'count_distribution_by_group.tsv', sep='\t', index=False)
    review = []
    for gene in genes:
        record = dict(annotations[gene])
        for window in SOURCES:
            r = effects[(effects.Geneid == gene) & (effects.window == window) & (effects.model == 'cell_adjusted')].iloc[0]
            q = loo[(loo.Geneid == gene) & (loo.window == window)]
            o = obs[(obs.Geneid == gene) & (obs.window == window)]
            for field in ['log2FoldChange', 'lfcSE', 'fold_change_PD_Control', 'FC_CI95_lower', 'FC_CI95_upper', 'padj']:
                record[f'adjusted_{window}_{field}'] = float(r[field])
            record[f'window_{window}_Cook_flagged_observations'] = int(o.cook_review_flag.sum())
            record[f'window_{window}_conditional_checks'] = len(q)
            record[f'window_{window}_maximum_abs_conditional_effect_shift'] = float(q.delta_log2FC.abs().max())
            record[f'window_{window}_maximum_abs_shift_SE_units'] = float(q.delta_original_SE_units.abs().max())
            record[f'window_{window}_all_checked_directions_retained'] = bool(q.same_direction.all())
        record['direction_matches_between_windows'] = bool(np.sign(record['adjusted_528_log2FoldChange']) == np.sign(record['adjusted_445_log2FoldChange']))
        review.append(record)
    reviewed = pd.DataFrame(review)
    reviewed.to_csv(ROOT / 'five_gene_review.tsv', sep='\t', index=False)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    rng = np.random.default_rng(20260912)
    for gene, ax in zip(genes, axes):
        r = obs[(obs.window == 528) & (obs.Geneid == gene)]
        data = [np.log2(r.loc[r.group == group, 'normalized_count'].to_numpy() + 1) for group in ['Control', 'PD']]
        ax.boxplot(data, positions=[0, 1], showfliers=False)
        for j, vals in enumerate(data):
            ax.scatter(j + rng.uniform(-.2, .2, len(vals)), vals, s=5, alpha=.3)
        ax.set_xticks([0, 1], ['Control', 'PD'])
        ax.set_title(annotations[gene]['reference_gene_name'])
    axes[0].set_ylabel('log2(size-factor-normalized count + 1)')
    fig.suptitle('528 participants: descriptive count distributions, without covariate adjustment')
    fig.tight_layout(); fig.savefig(ROOT / 'count_distributions.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
    labels = ['528 reference', '528 cell-adjusted', '445 reference', '445 cell-adjusted']
    for gene, ax in zip(genes, axes):
        r = effects[effects.Geneid == gene]
        for i, row in enumerate(r.itertuples(index=False)):
            ax.errorbar(row.log2FoldChange, i, xerr=1.959963984540054 * row.lfcSE, fmt='o', color=['#777777', '#1874b4', '#777777', '#cb661f'][i])
        ax.axvline(0, color='black', lw=.7)
        ax.set_title(annotations[gene]['reference_gene_name']); ax.set_xlabel('PD/control log2 fold change')
        ax.set_yticks(range(4), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Unshrunk effects and 95% Wald intervals; intervals do not adjust for selection')
    fig.tight_layout(); fig.savefig(ROOT / 'effect_intervals.png', dpi=180); plt.close(fig)
    assert all(sha(Path(path)) == value for path, value in hashes.items()), 'Source input changed'
    summary = {'status': 'COMPLETE', 'genes': annotations, 'exact_version_and_length_matches': 5,
               'conditional_checks': len(loo), 'conditional_checks_converged': int(loo.converged.sum()),
               'conditional_direction_reversals': int((~loo.same_direction).sum()),
               'maximum_absolute_conditional_log2FC_shift': float(loo.delta_log2FC.abs().max()),
               'maximum_absolute_conditional_shift_SE_units': float(loo.delta_original_SE_units.abs().max()),
               'Cook_flagged_gene_sample_observations': int(obs.cook_review_flag.sum()),
               'all_adjusted_directions_match_between_windows': bool(reviewed.direction_matches_between_windows.all()),
               'adjusted_528_FDR05': int(reviewed.adjusted_528_padj.lt(.05).sum()),
               'adjusted_445_FDR05': int(reviewed.adjusted_445_padj.lt(.05).sum()),
               'input_hashes_unchanged': True, 'compute': 'agpu_env Python and R CPU targeted review',
               'no_new_FDR_testing': True, 'no_full_normalization_dispersion_refit': True}
    save_json('summary.json', summary)
    lines = ['# Five-gene review', '', 'All five candidates match the original GENCODE v29 identifier/version and featureCounts exon-union length.', '',
             '| Gene | GENCODE v29 type | Adjusted FC: 528 | FDR: 528 | Adjusted FC: 445 | FDR: 445 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in reviewed.itertuples(index=False):
        lines.append(f'| {row.reference_gene_name} | {row.reference_gene_type} | {row.adjusted_528_fold_change_PD_Control:.3f} | {row.adjusted_528_padj:.4f} | {row.adjusted_445_fold_change_PD_Control:.3f} | {row.adjusted_445_padj:.4f} |')
    lines += ['', 'FC is PD/control; below 1 means lower expression in PD. Full unshrunk 95% intervals and all four models are in effect_estimates_all_four_models.tsv.', '',
              f"Selected conditional omission checks: {len(loo)} completed and converged; {summary['conditional_direction_reversals']} direction reversals. Maximum absolute shift was {summary['maximum_absolute_conditional_log2FC_shift']:.4f} log2 units ({summary['maximum_absolute_conditional_shift_SE_units']:.3f} original-SE units at the largest standardized shift).",
              f"Cook-threshold review flags: {summary['Cook_flagged_gene_sample_observations']} gene/sample observations across the two cell-adjusted models. All five adjusted directions match between windows: {summary['all_adjusted_directions_match_between_windows']}.", '',
              'These omissions target the maximum-Cook and maximum-normalized-count observations only. They hold size factors and dispersions fixed; they are not an exhaustive influence analysis, a full refit, or a new significance assessment.', '',
              'LINC01806 is a long noncoding RNA; the other four are annotated protein-coding. These biotypes constrain follow-up assay choices but do not establish assay specificity. Exact annotation is not proof of unique read mapping or functional mechanism.', '',
              'All five are exploratory candidates. None passes genome-wide FDR <0.05 in the tighter 445-person adjusted analysis. Direction agreement and selected conditional influence checks do not constitute external validation. The next evidence step is independent-cohort replication with prespecified targets and appropriate covariates.', '',
              '![Count distributions](count_distributions.png)', '', '![Effect intervals](effect_intervals.png)', '',
              'Input hashes, raw-count checks, exact identifiers, reference lengths, participant uniqueness and source-model dimensions were verified. No original result, cohort, annotation, or fitted object was modified.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    save_json('status.json', {'status': 'COMPLETE', 'report': 'RESULTS.md'})
    print(json.dumps({k: v for k, v in summary.items() if k != 'genes'}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save_json('status.json', {'status': 'FAILED', 'error': traceback.format_exc()})
        raise
