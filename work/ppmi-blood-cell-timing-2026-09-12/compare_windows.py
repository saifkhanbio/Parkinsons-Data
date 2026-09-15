"""Compare completed paired models between the nested 528 and 445 cohorts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'ppmi-blood-cell-adjustment-2026-09-12'
MODELS = ['subset_reference', 'cell_adjusted']


def compare_windows():
    from launch import device, save_json
    dev = device()
    old_cohort = pd.read_csv(PRIOR / 'blood_covariates.tsv', sep='\t', dtype={'PATNO': str})
    new_cohort = pd.read_csv(ROOT / 'blood_covariates.tsv', sep='\t', dtype={'PATNO': str})
    expected = old_cohort[old_cohort.month_gap_lab_minus_rna.isin([-1, 0])]
    assert new_cohort.PATNO.tolist() == expected.PATNO.tolist()
    fields = ['PATNO', 'group', 'panel_id', 'month_gap_lab_minus_rna', 'wbc',
              'neutrophils_percent', 'lymphocytes_percent', 'monocytes_percent',
              'eosinophils_percent', 'basophils_percent']
    pd.testing.assert_frame_equal(new_cohort[fields].reset_index(drop=True), expected[fields].reset_index(drop=True))
    comparison = pd.read_csv(ROOT / 'gene_comparison.tsv', sep='\t')[
        ['Geneid', 'gene_symbol', 'original_shortlist', 'original_seven_model_robust']]
    metrics = {}
    for label in MODELS:
        for n, directory in [(528, PRIOR), (445, ROOT)]:
            r = pd.read_csv(directory / label / 'results.tsv', sep='\t')
            cols = ['Geneid', 'log2FoldChange', 'lfcSE', 'padj', 'padj_no_independent_filter']
            comparison = comparison.merge(r[cols].rename(columns={c: f'{label}_{n}_{c}' for c in cols if c != 'Geneid'}), on='Geneid', validate='one_to_one')
        a, b = f'{label}_528_', f'{label}_445_'
        x, y = comparison[a + 'log2FoldChange'], comparison[b + 'log2FoldChange']
        finite = np.isfinite(x) & np.isfinite(y)
        pair = torch.tensor(np.column_stack([x[finite], y[finite]]), dtype=torch.float64, device=dev)
        corr = float(torch.corrcoef(pair.T)[0, 1].cpu())
        np.testing.assert_allclose(corr, np.corrcoef(x[finite], y[finite])[0, 1], atol=1e-12)
        sig_a = comparison[a + 'padj'].lt(.05)
        sig_b = comparison[b + 'padj'].lt(.05)
        same = np.sign(x) == np.sign(y)
        comparison[label + '_delta_445_minus_528_log2FC'] = y - x
        comparison[label + '_retained_528_significance_and_direction'] = sig_a & sig_b & same
        metrics[label] = {'effect_Pearson_528_vs_445': corr, 'significant_528': int(sig_a.sum()),
                          'significant_445': int(sig_b.sum()),
                          'retained_528_significance_and_direction': int((sig_a & sig_b & same).sum()),
                          'significant_direction_reversals': int((sig_a & sig_b & finite & ~same).sum()),
                          'median_abs_log2FC_shift_among_528_significant': float((y - x)[sig_a & finite].abs().median()),
                          'median_SE_ratio_445_over_528_among_528_significant': float((comparison[b + 'lfcSE'] / comparison[a + 'lfcSE'])[sig_a & finite].median())}
    assert len(comparison) == 21888
    for n in [528, 445]:
        comparison[f'cell_adjustment_{n}_delta_log2FC'] = comparison[f'cell_adjusted_{n}_log2FoldChange'] - comparison[f'subset_reference_{n}_log2FoldChange']
        comparison[f'cell_adjustment_{n}_delta_abs_log2FC'] = comparison[f'cell_adjusted_{n}_log2FoldChange'].abs() - comparison[f'subset_reference_{n}_log2FoldChange'].abs()
    comparison.to_csv(ROOT / 'timing_gene_comparison.tsv', sep='\t', index=False)
    for name, mask in [('shortlist_20', comparison.original_shortlist), ('robust_1852', comparison.original_seven_model_robust),
                       ('adjusted_528_five_genes', comparison.cell_adjusted_528_padj.lt(.05))]:
        comparison.loc[mask].to_csv(ROOT / f'timing_{name}_comparison.tsv', sep='\t', index=False)
    assert comparison.original_shortlist.sum() == 20
    assert comparison.original_seven_model_robust.sum() == 1852
    assert comparison.cell_adjusted_528_padj.lt(.05).sum() == 5
    paths = pd.read_csv(PRIOR / 'hallmark_comparison.tsv', sep='\t')[['pathway', 'original_robust_15']]
    for label in MODELS:
        for n, directory in [(528, PRIOR), (445, ROOT)]:
            h = pd.read_csv(directory / label / 'Hallmark_enrichment.tsv', sep='\t')
            paths = paths.merge(h[['pathway', 'NES', 'padj']].rename(columns={c: f'{label}_{n}_{c}' for c in ['NES', 'padj']}), on='pathway', validate='one_to_one')
        a, b = f'{label}_528_', f'{label}_445_'
        paths[label + '_retained_528_significance_and_direction'] = paths[a + 'padj'].lt(.05) & paths[b + 'padj'].lt(.05) & (np.sign(paths[a + 'NES']) == np.sign(paths[b + 'NES']))
        metrics[label]['Hallmark_retained_528_significance_and_direction'] = int(paths[label + '_retained_528_significance_and_direction'].sum())
        metrics[label]['original_15_retained_between_windows'] = int((paths.original_robust_15 & paths[label + '_retained_528_significance_and_direction']).sum())
    assert len(paths) == 50 and paths.original_robust_15.sum() == 15
    paths.to_csv(ROOT / 'timing_hallmark_comparison.tsv', sep='\t', index=False)
    paths[paths.original_robust_15].to_csv(ROOT / 'timing_robust_15_hallmark_comparison.tsv', sep='\t', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for label, ax in zip(MODELS, axes):
        x, y = comparison[f'{label}_528_log2FoldChange'], comparison[f'{label}_445_log2FoldChange']
        ax.scatter(x, y, s=3, alpha=.15, rasterized=True)
        limit = max(x.abs().max(), y.abs().max())
        ax.plot([-limit, limit], [-limit, limit], color='black', lw=.7)
        ax.set(title=label, xlabel='528-participant log2FC', ylabel='445-participant log2FC')
    fig.tight_layout(); fig.savefig(ROOT / 'timing_effect_comparison.png', dpi=180); plt.close(fig)
    save_json('timing_summary.json', {'cohort_validation': 'Exact nested subset and original CBC values verified',
                                      'n': 445, 'PD': 303, 'Control': 142, 'comparisons': metrics,
                                      'device': str(dev), 'independent_replication': False})
    lines = ['# Screening CBC timing sensitivity', '', '445 participants (303 PD, 142 controls): CBC in the RNA month or preceding month, compared with the 528-person main blood-cell analysis.', '',
             '| Model | Significant genes: 528 | Significant genes: 445 | Retained significance and direction | Effect correlation |',
             '| --- | ---: | ---: | ---: | ---: |']
    for label, s in metrics.items():
        lines.append(f"| {label} | {s['significant_528']} | {s['significant_445']} | {s['retained_528_significance_and_direction']} | {s['effect_Pearson_528_vs_445']:.4f} |")
    lines += ['', '## Interpretation checks', '']
    for label, s in metrics.items():
        lines.append(f"- {label}: among significant 528-person genes, median absolute effect shift is {s['median_abs_log2FC_shift_among_528_significant']:.4f} log2 units and median standard-error ratio (445/528) is {s['median_SE_ratio_445_over_528_among_528_significant']:.3f}. Of the original 15 Hallmark pathways, {s['original_15_retained_between_windows']} are significant with matching directions in both windows.")
    lines += ['', 'See RESULTS.md for within-445 paired results and the timing_* TSVs for full gene/pathway evidence, including the original shortlist, robust genes, and five adjusted-528 discoveries.', '',
              'This is a nested-cohort sensitivity, not external replication. The window changes both participants and timing. Smaller samples change uncertainty; threshold crossing alone does not show loss of a biological association. Neither window establishes same-day CBC/RNA collection. The 528-person paired analysis remains primary.', '']
    (ROOT / 'TIMING_RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    compare_windows()
