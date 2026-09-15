"""Validate full-refit QC sensitivity on RTX A3000 and produce comparisons."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parent / 'ppmi-deseq2-2026-09-12'
REVIEW = OUT.parent / 'ppmi-results-review-2026-09-12'
sys.path.insert(0, str(ANALYSIS))
from gpu_postprocess import bh, correlation
os.environ.setdefault('MPLCONFIGDIR', str(ANALYSIS / 'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def markdown(frame):
    def value(x):
        if pd.isna(x):
            return '—'
        return f'{x:.4g}' if isinstance(x, float) else str(x).replace('|', '/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |'] +
                     ['| '+' | '.join(value(x) for x in row)+' |'
                      for row in frame.itertuples(index=False, name=None)])


def main():
    assert torch.cuda.is_available(), 'CUDA required'
    devices = [i for i in range(torch.cuda.device_count()) if 'RTX A3000' in torch.cuda.get_device_name(i)]
    assert devices, 'Requested NVIDIA RTX A3000 unavailable'
    torch.cuda.set_device(devices[0])
    tensor = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), device='cuda', dtype=torch.float64)
    torch.testing.assert_close(bh(tensor([.01, .04, .03, .002])), tensor([.02, .04, .04, .008]))
    if '--gpu-check' in sys.argv:
        device = dict(status='VALIDATED',gpu=torch.cuda.get_device_name(torch.cuda.current_device()),
                      python=sys.executable,torch_version=torch.__version__,CUDA_BH_fixture='PASS')
        (OUT / 'gpu_preflight.json').write_text(json.dumps(device,indent=2)+'\n')
        print(json.dumps(device))
        return
    model = json.loads((OUT / 'model_summary.json').read_text())
    omitted_samples = json.loads((OUT / 'preflight.json').read_text())['omitted_samples']
    enrichment = json.loads((OUT / 'enrichment_summary.json').read_text())
    assert model['status'] == enrichment['status'] == 'COMPLETE'
    hashes = {}
    for row in pd.read_csv(OUT / 'input_md5.tsv', sep='\t').itertuples():
        data = Path(row.path).read_bytes()
        assert hashlib.md5(data).hexdigest() == row.md5
        hashes[row.path] = hashlib.sha256(data).hexdigest()
    current = pd.read_csv(OUT / 'results.tsv', sep='\t').set_index('Geneid')
    primary = pd.read_csv(ANALYSIS / 'primary/results_annotated.tsv', sep='\t').set_index('Geneid')
    assert current.index.is_unique and set(current.index) == set(primary.index)
    current = current.reindex(primary.index)
    assert current.loc[~current.beta_converged, ['stat','pvalue','padj']].isna().all().all()
    tested = current[current.padj.notna()]
    adjusted = bh(tensor(tested.pvalue))
    torch.testing.assert_close(adjusted, tensor(tested.padj), rtol=1e-8, atol=1e-12)
    errors = {'gene_BH_max_abs_error': float((adjusted-tensor(tested.padj)).abs().max())}

    # Independent CUDA implementation of DESeq2's ratio size-factor estimator.
    sf = pd.read_csv(OUT / 'size_factors.tsv', sep='\t', dtype={'PATNO':str}).set_index('PATNO')
    raw = pd.read_csv(OUT.parent / 'ppmi-expression-qc-2026-09-12/raw_counts.tsv.gz', sep='\t', index_col=0)
    frozen = pd.read_csv(OUT / 'frozen_genes.tsv', sep='\t').Geneid
    counts = tensor(raw.loc[frozen, sf.index].to_numpy())
    assert counts.shape == (21888, 556) and not set(sf.index) & set(map(str, omitted_samples))
    sums = pd.read_csv(OUT / 'count_sums.tsv', sep='\t', dtype={'PATNO':str}).set_index('PATNO').loc[sf.index,'count_sum']
    torch.testing.assert_close(counts.sum(0), tensor(sums), rtol=0, atol=0)
    positive = (counts > 0).all(1)
    logs = torch.log(counts[positive])
    expected_sf = torch.exp(torch.quantile(logs-logs.mean(1, keepdim=True), .5, dim=0))
    torch.testing.assert_close(expected_sf, tensor(sf.size_factor), rtol=1e-9, atol=1e-10)
    errors['size_factor_max_abs_error'] = float((expected_sf-tensor(sf.size_factor)).abs().max())
    prior_sf = pd.read_csv(ANALYSIS / 'primary/size_factors.tsv', sep='\t', dtype={'PATNO':str}).set_index('PATNO').loc[sf.index,'size_factor']
    ratio = np.log(sf.size_factor/prior_sf)
    sf['relative_change_after_common_scale_removal'] = np.exp(ratio-ratio.mean())
    sf.to_csv(OUT / 'normalization_comparison.tsv', sep='\t')
    del raw, counts, logs, expected_sf

    x, y = tensor(primary.log2FoldChange), tensor(current.log2FoldChange)
    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(tensor(primary.pvalue)) & torch.isfinite(tensor(current.pvalue))
    effect_correlation = correlation(x[valid], y[valid])
    np.testing.assert_allclose(effect_correlation, np.corrcoef(x[valid].cpu(),y[valid].cpu())[0,1], atol=1e-12)
    same = (torch.sign(x) == torch.sign(y)).cpu().numpy()
    comparison = primary[['gene_symbol','ensembl_id','log2FoldChange','lfcSE','padj','dispersion']].rename(
        columns={c:f'primary_{c}' for c in ['log2FoldChange','lfcSE','padj','dispersion']})
    comparison = comparison.join(current[['log2FoldChange','lfcSE','padj','dispersion','beta_converged']].rename(
        columns={c:f'QC_{c}' for c in ['log2FoldChange','lfcSE','padj','dispersion']}))
    comparison['same_direction'] = same
    comparison['delta_log2FC'] = (y-x).cpu().numpy()
    comparison['delta_in_primary_SE_units'] = comparison.delta_log2FC / primary.lfcSE
    comparison['significant_same_direction_primary_QC'] = primary.padj.lt(.05) & current.padj.lt(.05) & same
    original_robust = pd.read_csv(REVIEW / 'robust_gene_review.tsv', sep='\t').set_index('Geneid')
    six = original_robust[original_robust.significant_same_direction_all_six].index
    assert len(six) == 1852
    comparison['robust_prior_six'] = comparison.index.isin(six)
    comparison['robust_all_seven'] = comparison.robust_prior_six & comparison.significant_same_direction_primary_QC
    comparison['QC_fold_change_PD_Control'] = 2**comparison.QC_log2FoldChange
    comparison.to_csv(OUT / 'gene_comparison.tsv', sep='\t')
    comparison[comparison.robust_all_seven].sort_values('QC_padj').to_csv(OUT / 'robust_genes_all_seven.tsv', sep='\t')
    comparison[comparison.robust_prior_six & ~comparison.robust_all_seven].to_csv(OUT / 'previously_robust_not_retained.tsv', sep='\t')
    flagged_ids = pd.read_csv(REVIEW / 'flagged_gene_review.tsv', sep='\t').Geneid
    flagged = comparison.loc[flagged_ids]
    flagged.to_csv(OUT / 'flagged_gene_comparison.tsv', sep='\t')
    change = comparison.loc[primary.padj.lt(.05)].assign(abs_change=lambda z:z.delta_log2FC.abs()).sort_values('abs_change', ascending=False)
    change.head(50).to_csv(OUT / 'largest_changes_primary_significant.tsv', sep='\t')

    previous_paths = pd.read_csv(REVIEW / 'site_race_pathways/pathway_comparison.tsv', sep='\t')
    pathway_tables, pathway_summary = [], []
    for collection in ['Hallmark','C2','C5_BP']:
        z = pd.read_csv(OUT / f'{collection}_enrichment.tsv', sep='\t')
        observed = bh(tensor(z.pval))
        torch.testing.assert_close(observed, tensor(z.padj), rtol=1e-8, atol=1e-12)
        errors[f'{collection}_BH_max_abs_error'] = float((observed-tensor(z.padj)).abs().max())
        prior = previous_paths[previous_paths.collection == collection]
        z = z.rename(columns={c:f'{c}_QC' for c in ['pval','padj','NES','leadingEdge','size']})
        merged = prior.merge(z[['pathway','pval_QC','padj_QC','NES_QC','leadingEdge_QC','size_QC']],
                             on='pathway', how='outer', validate='one_to_one')
        merged['collection'] = collection
        same_path = np.sign(merged.NES_primary) == np.sign(merged.NES_QC)
        merged['retained_primary_QC'] = merged.padj_primary.lt(.05) & merged.padj_QC.lt(.05) & same_path
        merged['retained_primary_site_race_QC'] = merged.significant_same_direction_both.fillna(False).astype(bool) & merged.retained_primary_QC
        pathway_tables.append(merged)
        pathway_summary.append(dict(collection=collection,QC_tested=len(z),QC_significant=int(z.padj_QC.lt(.05).sum()),
            primary_significant=int(prior.padj_primary.lt(.05).sum()),
            retained_primary_QC=int(merged.retained_primary_QC.sum()),
            retained_primary_site_race_QC=int(merged.retained_primary_site_race_QC.sum()),
            significant_reversals_primary_QC=int((merged.padj_primary.lt(.05)&merged.padj_QC.lt(.05)&~same_path).sum())))
    paths = pd.concat(pathway_tables, ignore_index=True)
    joint = paths.collection.isin(['C2','C5_BP']) & paths.pval_QC.notna()
    paths.loc[joint,'padj_joint_C2_C5_BP_QC'] = bh(tensor(paths.loc[joint,'pval_QC'])).cpu().numpy()
    paths.to_csv(OUT / 'pathway_comparison.tsv', sep='\t', index=False)
    h = pd.read_csv(REVIEW / 'site_race_pathways/hallmark_six_model_comparison.tsv', sep='\t')
    h = h.merge(paths[paths.collection == 'Hallmark'][['pathway','NES_QC','padj_QC']], on='pathway', how='left', validate='one_to_one')
    h['significant_same_direction_all_seven'] = (h.significant_same_direction_all_six & h.padj_QC.lt(.05) &
        (np.sign(h.primary_NES) == np.sign(h.NES_QC)))
    h.to_csv(OUT / 'hallmark_seven_model_comparison.tsv', sep='\t', index=False)
    table = pd.DataFrame(pathway_summary)
    table.to_csv(OUT / 'pathway_summary.tsv', sep='\t', index=False)
    metrics = dict(status='COMPLETE',gpu=torch.cuda.get_device_name(torch.cuda.current_device()),
        QC_samples=556,omitted_samples=omitted_samples,QC_significant=model['significant_FDR05'],
        QC_nonconverged=model['nonconverged'],primary_QC_log2FC_correlation=effect_correlation,
        primary_significant_retained=int(comparison.significant_same_direction_primary_QC.sum()),
        prior_1852_robust_retained=int(comparison.robust_all_seven.sum()),
        original_15_hallmark_retained_all_seven=int(h.significant_same_direction_all_seven.sum()),
        primary_significant_direction_reversals=int((primary.padj.lt(.05)&current.padj.lt(.05)&~same).sum()),
        median_abs_log2FC_change_primary_significant=float(change.abs_change.median()),
        max_abs_log2FC_change_primary_significant=float(change.abs_change.max()),
        validation=errors,pathways=pathway_summary,input_sha256=hashes)

    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    finite = valid.cpu().numpy()
    axes[0].scatter(primary.loc[finite,'log2FoldChange'],current.loc[finite,'log2FoldChange'],s=3,alpha=.25,color='#287F9C')
    lo = min(primary.loc[finite,'log2FoldChange'].min(),current.loc[finite,'log2FoldChange'].min())
    hi = max(primary.loc[finite,'log2FoldChange'].max(),current.loc[finite,'log2FoldChange'].max())
    axes[0].plot([lo,hi],[lo,hi],color='grey',lw=1)
    axes[0].set(xlabel='Primary log2 fold change',ylabel='QC sensitivity log2 fold change',title=f'Effect agreement: r={effect_correlation:.4f}')
    axes[1].hist(change.delta_log2FC.dropna(),bins=60,color='#287F9C')
    axes[1].set(xlabel='QC minus primary log2 fold change',ylabel='Genes',title='Original primary significant genes')
    fig.savefig(OUT / 'effect_comparison.png',dpi=180)
    fig.savefig(OUT / 'effect_comparison.pdf')
    plt.close(fig)
    selected_h = h[h.significant_same_direction_all_six]
    report = '# Local analysis report\n\nConsult the locally generated tables and diagnostics.\n'
    (OUT / 'README.md').write_text(report)
    parent = REVIEW / 'README.md'
    text = parent.read_text()
    link = '../ppmi-qc-sensitivity-2026-09-12/README.md'
    if link not in text:
        parent.write_text(text + f'\n## Full-refit sample-quality sensitivity\n\n[Joint omission of the selected QC samples]({link}) is complete: {metrics["prior_1852_robust_retained"]:,}/1,852 prior robust genes and {metrics["original_15_hallmark_retained_all_seven"]}/15 Hallmark pathways retained across seven models.\n')
    (OUT / 'summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
