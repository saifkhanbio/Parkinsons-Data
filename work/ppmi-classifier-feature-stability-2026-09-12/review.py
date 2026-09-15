"""Review retained RNA features without refitting any classifier."""
from pathlib import Path
import hashlib
import itertools
import json
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'ppmi-classifier-refinement-retry-2026-09-12'
ANNOTATION = ROOT.parent / 'ppmi-gene-annotation-review-2026-09-12/gencode_v29_gene_reference.tsv'
BASE = ROOT.parent / 'ppmi-classifier-2026-09-12'
KEYS = ['protocol', 'family', 'penalty']
FIT = KEYS + ['repeat', 'fold']
PROTOCOLS = ['participant_stratified', 'batch_grouped']
FAMILIES = ['rna', 'combined']
PENALTIES = ['ridge', 'elasticnet']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(name, table):
    table.to_csv(ROOT / name, sep='\t', index=False)


def main():
    assert not (ROOT / 'summary.json').exists(), 'Review already exists; preserve prior outputs.'
    paths = [SOURCE / 'outer_gene_coefficients.tsv', SOURCE / 'outer_selected_parameters.tsv',
             SOURCE / 'summary.json', SOURCE / 'input_sha256.json', ANNOTATION,
             BASE / 'gene_ids.npy', ROOT / 'README.md', ROOT / 'review.py']
    hashes = {str(p): sha(p) for p in paths}
    assert json.loads((SOURCE / 'summary.json').read_text())['status'] == 'COMPLETE'
    saved_hashes = json.loads((SOURCE / 'input_sha256.json').read_text())
    for path, value in saved_hashes.items():
        assert sha(Path(path)) == value, f'Original refinement input changed: {path}'
    df = pd.read_csv(paths[0], sep='\t')
    parameters = pd.read_csv(paths[1], sep='\t')
    parameters = parameters[parameters.family.isin(FAMILIES)].copy()
    assert len(parameters) == 120 and not parameters.duplicated(FIT).any()
    assert not df.duplicated(FIT + ['Geneid']).any() and np.isfinite(df.coefficient).all()
    assert np.array_equal(df.nonzero.to_numpy(), df.coefficient.ne(0).to_numpy())
    universe = np.load(BASE / 'gene_ids.npy')
    genes = np.sort(df.Geneid.unique())
    assert set(genes) <= set(universe)
    annotation = pd.read_csv(ANNOTATION, sep='\t').rename(columns={'reference_Geneid': 'Geneid'})
    assert annotation.Geneid.is_unique
    annotation = annotation[['Geneid', 'reference_gene_name', 'reference_gene_type']]
    assert set(genes) <= set(annotation.Geneid)
    panels = df.groupby(FIT).agg(screened=('Geneid', 'size'), retained=('nonzero', 'sum')).reset_index()
    panels = parameters.merge(panels, on=FIT, validate='one_to_one')
    assert len(panels) == 120 and (panels.screened == panels.k).all()
    assert (panels.total_features == panels.k + np.where(panels.family.eq('combined'), 7, 0)).all()
    assert (panels.retained <= panels.nonzero_coefficients).all()
    assert ((panels.nonzero_coefficients-panels.retained) <= np.where(panels.family.eq('combined'), 7, 0)).all()
    write('fit_panel_sizes.tsv', panels)
    columns = pd.MultiIndex.from_product([[1, 2, 3], [1, 2, 3, 4, 5]], names=['repeat', 'fold'])
    tables, repeat_tables, pairwise = [], [], []
    for protocol, family, penalty in itertools.product(PROTOCOLS, FAMILIES, PENALTIES):
        sub = df[df.protocol.eq(protocol) & df.family.eq(family) & df.penalty.eq(penalty)]
        assert len(sub[['repeat', 'fold']].drop_duplicates()) == 15
        pivot = sub.pivot(index='Geneid', columns=['repeat', 'fold'], values='coefficient').reindex(index=genes, columns=columns)
        screened = pivot.notna().to_numpy()
        weights = pivot.fillna(0).to_numpy()
        kept = weights != 0
        substantial = np.abs(weights) >= 1e-4
        count = kept.sum(1)
        pos, neg = (weights > 0).sum(1), (weights < 0).sum(1)
        consistency = np.divide(np.maximum(pos, neg), count, out=np.zeros(len(genes)), where=count > 0)
        repeat_counts = kept.reshape(len(genes), 3, 5).sum(2)
        tiny_repeat_counts = substantial.reshape(len(genes), 3, 5).sum(2)
        tiny_pos, tiny_neg = ((weights >= 1e-4).sum(1)), ((weights <= -1e-4).sum(1))
        tiny_consistency = np.divide(np.maximum(tiny_pos, tiny_neg), substantial.sum(1),
                                     out=np.zeros(len(genes)), where=substantial.sum(1)>0)
        stable = (count >= 12) & (repeat_counts.min(1) >= 4) & (consistency >= .9)
        table = pd.DataFrame(dict(Geneid=genes, protocol=protocol, family=family, penalty=penalty,
                                  fits_total=15, screened_fits=screened.sum(1), retained_fits=count,
                                  retained_fraction=count / 15, positive_fits=pos, negative_fits=neg,
                                  sign_consistency=consistency, predominant_sign=np.sign(pos-neg).astype(int),
                                  repeat_1_retained=repeat_counts[:, 0], repeat_2_retained=repeat_counts[:, 1],
                                  repeat_3_retained=repeat_counts[:, 2], repeats_with_any_retention=(repeat_counts>0).sum(1),
                                  minimum_retained_in_any_repeat=repeat_counts.min(1),
                                  mean_coefficient_all_fits=weights.mean(1), median_coefficient_all_fits=np.median(weights, axis=1),
                                  mean_absolute_coefficient_all_fits=np.abs(weights).mean(1),
                                  mean_coefficient_when_retained=np.divide(weights.sum(1), count, out=np.full(len(genes), np.nan), where=count>0),
                                  minimum_coefficient_all_fits=weights.min(1), maximum_coefficient_all_fits=weights.max(1),
                                  stable=stable, retained_fits_abs_ge_1e_minus4=substantial.sum(1),
                                  stable_abs_ge_1e_minus4=(substantial.sum(1)>=12) & (tiny_repeat_counts.min(1)>=4) & (tiny_consistency>=.9)))
        assert (table.retained_fits <= table.screened_fits).all() and (table.screened_fits <= 15).all()
        tables.append(table)
        for repeat in [1, 2, 3]:
            repeat_tables.append(pd.DataFrame(dict(Geneid=genes, protocol=protocol, family=family,
                                                   penalty=penalty, repeat=repeat, folds_total=5,
                                                   retained_folds=repeat_counts[:, repeat-1])))
        for a, b in itertools.combinations(range(15), 2):
            union = int((kept[:, a] | kept[:, b]).sum())
            intersection = int((kept[:, a] & kept[:, b]).sum())
            pairwise.append(dict(protocol=protocol, family=family, penalty=penalty,
                                  repeat_a=columns[a][0], fold_a=columns[a][1], repeat_b=columns[b][0], fold_b=columns[b][1],
                                  comparison='within_repeat' if columns[a][0]==columns[b][0] else 'between_repeats',
                                  size_a=int(kept[:, a].sum()), size_b=int(kept[:, b].sum()), intersection=intersection,
                                  union=union, jaccard=intersection/union if union else np.nan))
    stability = pd.concat(tables, ignore_index=True).merge(annotation, on='Geneid', validate='many_to_one')
    assert len(stability) == 8 * len(genes)
    write('annotated_stability.tsv', stability)
    write('per_repeat_retention.tsv', pd.concat(repeat_tables, ignore_index=True))
    pairs = pd.DataFrame(pairwise)
    assert len(pairs) == 8 * 105
    write('pairwise_jaccard.tsv', pairs)
    overlaps = pairs.groupby(KEYS + ['comparison']).jaccard.agg(['median', 'min', 'max']).reset_index()
    write('jaccard_summary.tsv', overlaps)
    cross = []
    for family, penalty in itertools.product(FAMILIES, PENALTIES):
        part = stability[stability.family.eq(family) & stability.penalty.eq(penalty)]
        a = part[part.protocol.eq(PROTOCOLS[0])].set_index('Geneid').loc[genes]
        b = part[part.protocol.eq(PROTOCOLS[1])].set_index('Geneid').loc[genes]
        same_sign = a.predominant_sign.eq(b.predominant_sign) & a.predominant_sign.ne(0)
        cross.append(pd.DataFrame(dict(Geneid=genes, family=family, penalty=penalty,
                     participant_retained=a.retained_fits.to_numpy(), batch_retained=b.retained_fits.to_numpy(),
                     minimum_retained=np.minimum(a.retained_fits, b.retained_fits).to_numpy(),
                     same_sign=same_sign.to_numpy(), predominant_sign=a.predominant_sign.to_numpy(),
                     stable_both=(a.stable & b.stable & same_sign).to_numpy(),
                     stable_both_abs_ge_1e_minus4=(a.stable_abs_ge_1e_minus4 & b.stable_abs_ge_1e_minus4 & same_sign).to_numpy())))
    cross = pd.concat(cross, ignore_index=True).merge(annotation, on='Geneid', validate='many_to_one')
    write('cross_protocol_stability.tsv', cross)
    en = stability[stability.penalty.eq('elasticnet')]
    ranking = en.groupby('Geneid').agg(minimum_retained_across_four=('retained_fits', 'min'),
             total_retained_across_four=('retained_fits', 'sum'), minimum_sign_consistency=('sign_consistency', 'min'),
             minimum_per_repeat=('minimum_retained_in_any_repeat', 'min'), stable_strata=('stable', 'sum'),
             stable_strata_abs_ge_1e_minus4=('stable_abs_ge_1e_minus4', 'sum'), minimum_sign=('predominant_sign', 'min'),
             maximum_sign=('predominant_sign', 'max'), mean_absolute_coefficient=('mean_absolute_coefficient_all_fits', 'mean')).reset_index()
    ranking['consistent_sign_across_four'] = ranking.minimum_sign.eq(ranking.maximum_sign) & ranking.minimum_sign.ne(0)
    ranking['strict_core'] = ranking.stable_strata.eq(4) & ranking.consistent_sign_across_four
    ranking['strict_core_abs_ge_1e_minus4'] = ranking.stable_strata_abs_ge_1e_minus4.eq(4) & ranking.consistent_sign_across_four
    ranking = ranking.merge(annotation, on='Geneid', validate='one_to_one').sort_values(
        ['strict_core', 'minimum_retained_across_four', 'minimum_sign_consistency', 'total_retained_across_four', 'mean_absolute_coefficient', 'Geneid'],
        ascending=[False, False, False, False, False, True])
    for protocol, family in itertools.product(PROTOCOLS, FAMILIES):
        part = en[en.protocol.eq(protocol) & en.family.eq(family)].set_index('Geneid')
        ranking[f'{protocol}_{family}_retained'] = ranking.Geneid.map(part.retained_fits)
        ranking[f'{protocol}_{family}_sign_consistency'] = ranking.Geneid.map(part.sign_consistency)
    write('elasticnet_consensus_ranking.tsv', ranking)
    core = ranking[ranking.strict_core]
    write('elasticnet_core.tsv', core)
    counts = stability.groupby(KEYS).agg(ever_screened=('screened_fits', lambda x: int((x>0).sum())),
                 ever_retained=('retained_fits', lambda x: int((x>0).sum())), stable=('stable', 'sum'),
                 stable_abs_ge_1e_minus4=('stable_abs_ge_1e_minus4', 'sum')).reset_index()
    write('stable_gene_counts.tsv', counts)
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top = ranking.head(20)
    plot_columns = list(itertools.product(PROTOCOLS, FAMILIES, ['elasticnet', 'ridge']))
    matrix = np.array([stability[stability.protocol.eq(p) & stability.family.eq(f) & stability.penalty.eq(q)].set_index('Geneid').loc[top.Geneid, 'retained_fits'].to_numpy() for p, f, q in plot_columns]).T
    fig, ax = plt.subplots(figsize=(12, max(5, .3*len(top)+2)))
    heat = ax.imshow(matrix, vmin=0, vmax=15, cmap='YlGnBu', aspect='auto')
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f'{r.reference_gene_name} [{r.Geneid}]' for r in top.itertuples(index=False)], fontsize=8)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([f'{"Participant" if p==PROTOCOLS[0] else "Batch"}\n{f}\n{q}' for p, f, q in plot_columns], fontsize=8)
    for i, j in itertools.product(range(len(top)), range(8)):
        ax.text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=8, color='white' if matrix[i,j]>=10 else 'black')
    ax.set_title('Top 20 by elastic-net recurrence: nonzero retention out of 15 fits')
    fig.colorbar(heat, ax=ax, label='Fits retaining gene (out of 15)')
    fig.tight_layout(); fig.savefig(ROOT / 'stability_heatmap.png', dpi=180); plt.close(fig)
    summary = dict(status='COMPLETE', fits_per_stratum=15, strata=8, RNA_bearing_fits=120,
                   distinct_genes_ever_screened=len(genes), genes_never_screened=len(universe)-len(genes),
                   exact_annotation_matches=len(genes), strict_elasticnet_core=len(core),
                   strict_core_abs_ge_1e_minus4=int(ranking.strict_core_abs_ge_1e_minus4.sum()),
                   stability_counts=counts.to_dict(orient='records'),
                   stable_across_protocols=cross.groupby(['family', 'penalty']).stable_both.sum().reset_index().to_dict(orient='records'),
                   validations='PASS', no_new_models_fitted=True, external_validation=False)
    lines = ['# Classifier RNA feature stability', '',
             'Review of 120 existing outer-trained RNA/combined fits: 15 fits in each protocol/family/penalty stratum. All fitted participants are from the same 528-person PPMI cohort.', '',
             '**Stable** means nonzero in at least 12/15 fits, at least 4/5 folds in every repeat, and at least 90% coefficient-sign consistency. A gene merely entering the screened RNA matrix does not count as retained when elastic-net sets its coefficient to zero.', '',
             f'The strict elastic-net core contains **{len(core)} genes**, stable across both split protocols and both RNA/combined models, with the same predominant coefficient sign. **{int(ranking.strict_core_abs_ge_1e_minus4.sum())}** pass when coefficients smaller than 1e-4 are treated as zero.', '',
             '| Protocol | Model | Penalty | Ever retained | Stable genes |', '| --- | --- | --- | ---: | ---: |']
    for r in counts.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.family} | {r.penalty} | {r.ever_retained} | {r.stable} |')
    lines += ['', '## Leading genes by elastic-net recurrence', '',
              'Counts are nonzero fits out of 15. Direction is a conditional classifier coefficient, not a differential-expression result. Entries outside the strict core, if any, are explicitly marked.', '',
              '| GENCODE v29 name | Versioned gene ID | Participant RNA | Participant combined | Batch RNA | Batch combined | Direction | Strict core |',
              '| --- | --- | ---: | ---: | ---: | ---: | --- | --- |']
    for r in top.itertuples(index=False):
        direction = ('Positive' if r.minimum_sign>0 else 'Negative') if r.consistent_sign_across_four else 'Mixed/absent'
        lines.append(f'| {r.reference_gene_name} | {r.Geneid} | {r.participant_stratified_rna_retained} | {r.participant_stratified_combined_retained} | {r.batch_grouped_rna_retained} | {r.batch_grouped_combined_retained} | {direction} | {"Yes" if r.strict_core else "No"} |')
    lines += ['', '## Interpretation', '',
              'Ridge retention largely measures recurrence in the top-k screen; elastic-net also tests survival of its sparsity penalty. All stability denominators include fits where a gene was absent. Ranking prioritizes worst-stratum elastic-net retention and sign consistency, not validation performance.', '',
              'These overlapping folds and repeated partitions do not establish independent replication. Conditional coefficient signs can change as correlated predictors exchange weights. Retention does not prove independent RNA value beyond blood counts or a gene-specific causal effect. The modest classifier discrimination remains unchanged.', '',
              'The consensus list was assembled after viewing development results. It is a follow-up list, not a validated reduced classifier; existing cross-validation AUROCs cannot be assigned to a new model fitted using this list. Exact gene versions are the keys; reference names/biotypes are GENCODE v29 labels.', '',
              'All fit-key, feature-total, retention, annotation, and source-integrity validations passed. Full per-gene/per-repeat metrics, panel sizes, and Jaccard overlaps are saved alongside this report.', '',
              '![Retention across models](stability_heatmap.png)', '']
    assert all(sha(Path(path)) == expected for path, expected in hashes.items())
    (ROOT / 'input_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
