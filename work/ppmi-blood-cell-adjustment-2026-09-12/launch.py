"""Prepare, launch without polling, and automatically finalize the paired analysis."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BASE = WORK / 'ppmi-deseq2-2026-09-12'
FROZEN = WORK / 'ppmi-blood-counts-2026-09-12/approved_cohort_528'
MODELS = ['subset_reference', 'cell_adjusted']


def save_json(name, data):
    target = ROOT / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(target)


def status(state, **kwargs):
    save_json('status.json', {'status': state, 'utc': datetime.now(timezone.utc).isoformat(), **kwargs})


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def device():
    if torch.cuda.is_available():
        ids = [i for i in range(torch.cuda.device_count()) if 'RTX' in torch.cuda.get_device_name(i)]
        return torch.device('cuda', ids[0] if ids else 0)
    return torch.device('cpu')


def prepare():
    assert not (ROOT / 'status.json').exists(), 'Existing run: inspect status; do not relaunch into this directory.'
    m = pd.read_csv(FROZEN / 'cohort_manifest.tsv', sep='\t', dtype={'PATNO': str})
    d = pd.read_csv(BASE / 'primary_colData.tsv', sep='\t', dtype={'PATNO': str})
    assert len(m) == 528 and m.PATNO.is_unique and d.PATNO.is_unique
    assert m.group.value_counts().to_dict() == {'PD': 358, 'Control': 170}
    assert m.month_gap_lab_minus_rna.isin([-2, -1, 0]).all()
    paired = d.set_index('PATNO').loc[m.PATNO]
    assert (paired.group.to_numpy() == m.group.to_numpy()).all()
    scales = {}
    for output, source in [('log_wbc_z', 'wbc'), ('neutrophils_z', 'neutrophils_percent'),
                           ('monocytes_z', 'monocytes_percent'), ('eosinophils_z', 'eosinophils_percent'),
                           ('basophils_z', 'basophils_percent')]:
        values = m[source].to_numpy(float)
        if source == 'wbc':
            assert (values > 0).all()
            values = np.log(values)
        assert np.isfinite(values).all() and values.std(ddof=1) > 0
        scales[output] = {'source': source, 'transform': 'natural_log' if source == 'wbc' else 'identity',
                          'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))}
        m[output] = (values - values.mean()) / values.std(ddof=1)
    m.to_csv(ROOT / 'blood_covariates.tsv', sep='\t', index=False)
    save_json('cell_covariate_scaling.json', scales)
    balance = []
    for group in ['PD', 'Control']:
        for label, subset in [('retained', d[d.PATNO.isin(m.PATNO)]), ('excluded', d[~d.PATNO.isin(m.PATNO)])]:
            sub = subset[subset.group == group]
            record = {'group': group, 'subset': label, 'n': len(sub), 'female_n': int(sub.sex.eq('Female').sum())}
            for col in ['age_collection_years', 'RIN', 'intergenic_percent']:
                record[col + '_mean'] = sub[col].mean()
                record[col + '_sd'] = sub[col].std()
            balance.append(record)
    pd.DataFrame(balance).to_csv(ROOT / 'subset_balance.tsv', sep='\t', index=False)
    with (ROOT / 'prepare.log').open('w') as log:
        subprocess.run(['/usr/local/bin/Rscript', str(ROOT / 'run_models.R'), '--prepare-only'],
                       stdout=log, stderr=subprocess.STDOUT, check=True, cwd=WORK.parent)
    dev = device()
    validation = {'device': str(dev), 'device_name': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'CPU fallback',
                  'python': sys.executable, 'torch': torch.__version__, 'models': {}}
    for label in MODELS:
        table = pd.read_csv(ROOT / label / 'design_matrix.tsv', sep='\t', dtype={'PATNO': str})
        assert table.PATNO.tolist() == m.PATNO.tolist()
        x = table.drop(columns='PATNO').to_numpy(float)
        scaled = x / np.linalg.norm(x, axis=0)
        cpu = np.linalg.svd(scaled, compute_uv=False)
        gpu = torch.linalg.svdvals(torch.tensor(scaled, dtype=torch.float64, device=dev)).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu, rtol=1e-10, atol=1e-12)
        rank = int((gpu > gpu[0] * max(x.shape) * np.finfo(float).eps).sum())
        assert rank == x.shape[1]
        validation['models'][label] = {'rank': rank, 'columns': x.shape[1], 'n': len(x),
                                       'unit_column_norm_condition_number': float(gpu[0] / gpu[-1]),
                                       'CPU_GPU_singular_values_agree': True}
    save_json('compute_validation.json', validation)
    inputs = [FROZEN / 'cohort_manifest.tsv', FROZEN / 'summary.json', BASE / 'primary/dds.rds',
              BASE / 'primary/results_annotated.tsv', BASE / 'primary/hallmark_enrichment.tsv',
              BASE / 'primary_colData.tsv', BASE / 'msigdb_hallmark_gobp.rds',
              WORK / 'ppmi-gene-prioritization-2026-09-12/shortlist_20.tsv',
              WORK / 'ppmi-qc-sensitivity-2026-09-12/robust_genes_all_seven.tsv',
              WORK / 'ppmi-qc-sensitivity-2026-09-12/hallmark_seven_model_comparison.tsv',
              ROOT / 'run_models.R', ROOT / 'launch.py', ROOT / 'blood_covariates.tsv']
    save_json('input_sha256.json', {str(p): sha(p) for p in inputs})
    return validation


def bh(values, dev):
    p = np.asarray(values, dtype=float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    if not len(p):
        return p
    t = torch.tensor(p, dtype=torch.float64, device=dev)
    order = torch.argsort(t)
    ordered = t[order] * len(t) / torch.arange(1, len(t) + 1, device=dev, dtype=torch.float64)
    adjusted = torch.cummin(ordered.flip(0), dim=0).values.flip(0).clamp(max=1)
    answer = torch.empty_like(t); answer[order] = adjusted
    answer = answer.cpu().numpy()
    ix = np.argsort(p)
    cpu = np.empty_like(p)
    cpu[ix] = np.minimum(1, np.minimum.accumulate((p[ix] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    np.testing.assert_allclose(answer, cpu, rtol=1e-10, atol=1e-12)
    return answer


def finalize():
    dev = device()
    models = {label: pd.read_csv(ROOT / label / 'results.tsv', sep='\t') for label in MODELS}
    original = pd.read_csv(BASE / 'primary/results_annotated.tsv', sep='\t')
    comparison = original[['Geneid', 'gene_symbol', 'log2FoldChange', 'lfcSE', 'padj']].rename(
        columns={k: 'primary_558_' + k for k in ['log2FoldChange', 'lfcSE', 'padj']})
    validation = {}
    sf = []
    gene_order = pd.read_csv(ROOT / 'frozen_genes.tsv', sep='\t').Geneid.tolist()
    for label, r in models.items():
        assert r.Geneid.is_unique and r.Geneid.tolist() == gene_order and len(r) == 21888
        for col in ['padj', 'padj_no_independent_filter']:
            mask = r[col].notna()
            np.testing.assert_allclose(bh(r.loc[mask, 'pvalue'], dev), r.loc[mask, col], rtol=1e-8, atol=1e-12)
        assert r.loc[~r.beta_converged, ['pvalue', 'padj', 'stat']].isna().all().all()
        keep = ['Geneid', 'log2FoldChange', 'lfcSE', 'padj', 'padj_no_independent_filter', 'cook_review_flag']
        comparison = comparison.merge(r[keep].rename(columns={k: label + '_' + k for k in keep if k != 'Geneid'}), on='Geneid', validate='one_to_one')
        sf.append(pd.read_csv(ROOT / label / 'size_factors.tsv', sep='\t').size_factor.to_numpy())
        validation[label] = {'BH_checks': 'PASS', 'genes': len(r)}
    np.testing.assert_allclose(sf[0], sf[1], rtol=1e-12, atol=1e-12)
    metrics = {}
    for label, a, b in [('subset_effect', 'primary_558', 'subset_reference'), ('cell_adjustment_effect', 'subset_reference', 'cell_adjusted')]:
        xa, xb = comparison[a + '_log2FoldChange'], comparison[b + '_log2FoldChange']
        finite = xa.notna() & xb.notna()
        t = torch.tensor(np.column_stack([xa[finite], xb[finite]]), dtype=torch.float64, device=dev)
        corr = float(torch.corrcoef(t.T)[0, 1].cpu())
        np.testing.assert_allclose(corr, np.corrcoef(xa[finite], xb[finite])[0, 1], atol=1e-12)
        comparison[label + '_delta_log2FC'] = xb - xa
        sig_a, sig_b = comparison[a + '_padj'].lt(.05), comparison[b + '_padj'].lt(.05)
        same = np.sign(xa) == np.sign(xb)
        comparison[label + '_retained_significance_and_direction'] = sig_a & sig_b & same
        metrics[label] = {'log2FC_Pearson': corr, 'reference_significant': int(sig_a.sum()),
                          'retained_significant_same_direction': int((sig_a & sig_b & same).sum()),
                          'significant_opposite_directions': int((sig_a & sig_b & ~same).sum()),
                          'median_abs_log2FC_shift_reference_significant': float((xb - xa)[sig_a & finite].abs().median())}
    shortlist = pd.read_csv(WORK / 'ppmi-gene-prioritization-2026-09-12/shortlist_20.tsv', sep='\t')
    robust = pd.read_csv(WORK / 'ppmi-qc-sensitivity-2026-09-12/robust_genes_all_seven.tsv', sep='\t')
    assert len(shortlist) == 20 and len(robust) == 1852
    comparison['original_shortlist'] = comparison.Geneid.isin(shortlist.Geneid)
    comparison['original_seven_model_robust'] = comparison.Geneid.isin(robust.Geneid)
    comparison.to_csv(ROOT / 'gene_comparison.tsv', sep='\t', index=False)
    selected = shortlist[['Geneid']].merge(comparison, on='Geneid', validate='one_to_one')
    selected.to_csv(ROOT / 'shortlist_20_comparison.tsv', sep='\t', index=False)
    robust_result = comparison[comparison.original_seven_model_robust]
    assert len(robust_result) == 1852
    robust_result.to_csv(ROOT / 'robust_1852_comparison.tsv', sep='\t', index=False)
    for name, r in [('shortlist_20', selected), ('robust_1852', robust_result)]:
        metrics[name] = {'n': len(r)}
        for label in MODELS:
            metrics[name][label + '_retained_primary_direction_FDR05'] = int((r[label + '_padj'].lt(.05) & (np.sign(r[label + '_log2FoldChange']) == np.sign(r.primary_558_log2FoldChange))).sum())
    paths = pd.read_csv(BASE / 'primary/hallmark_enrichment.tsv', sep='\t')[['pathway', 'NES', 'padj']].rename(columns={'NES': 'primary_558_NES', 'padj': 'primary_558_padj'})
    robust_h = pd.read_csv(WORK / 'ppmi-qc-sensitivity-2026-09-12/hallmark_seven_model_comparison.tsv', sep='\t')
    robust_names = robust_h.loc[robust_h.significant_same_direction_all_seven, 'pathway']
    assert len(robust_names) == 15
    for label in MODELS:
        h = pd.read_csv(ROOT / label / 'Hallmark_enrichment.tsv', sep='\t')
        np.testing.assert_allclose(bh(h.pval, dev), h.padj, rtol=1e-8, atol=1e-12)
        paths = paths.merge(h[['pathway', 'NES', 'padj']].rename(columns={'NES': label + '_NES', 'padj': label + '_padj'}), on='pathway', validate='one_to_one')
    assert len(paths) == 50
    paths['original_robust_15'] = paths.pathway.isin(robust_names)
    paths.to_csv(ROOT / 'hallmark_comparison.tsv', sep='\t', index=False)
    paths[paths.original_robust_15].to_csv(ROOT / 'robust_15_hallmark_comparison.tsv', sep='\t', index=False)
    metrics['robust_15_hallmark'] = {label + '_retained_primary_direction_FDR05': int((paths.original_robust_15 & paths[label + '_padj'].lt(.05) & (np.sign(paths[label + '_NES']) == np.sign(paths.primary_558_NES))).sum()) for label in MODELS}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, a, b) in zip(axes, [('Subset effect', 'primary_558', 'subset_reference'), ('Measured cell adjustment', 'subset_reference', 'cell_adjusted')]):
        x, y = comparison[a + '_log2FoldChange'], comparison[b + '_log2FoldChange']
        ax.scatter(x, y, s=3, alpha=.15, rasterized=True)
        lim = max(x.abs().max(), y.abs().max())
        ax.plot([-lim, lim], [-lim, lim], color='black', lw=.7)
        ax.set(xlabel=a + ' log2FC', ylabel=b + ' log2FC', title=title)
    fig.tight_layout(); fig.savefig(ROOT / 'effect_comparison.png', dpi=180); plt.close(fig)
    sizes = {label: json.loads((ROOT / label / 'model_summary.json').read_text()) for label in MODELS}
    save_json('analysis_summary.json', {'models': sizes, 'comparisons': metrics, 'validation': validation,
                                     'same_count_normalization_agrees': True, 'device': str(dev)})
    lines = ['# PPMI measured blood-cell adjustment results', '',
             '528 participants: 358 PD and 170 controls. Screening CBCs are from the RNA month or either of the two preceding calendar months.', '',
             '| Model | Significant genes, FDR <0.05 | Higher PD | Lower PD | Nonconverged |', '| --- | ---: | ---: | ---: | ---: |']
    for label, s in sizes.items():
        lines.append(f"| {label} | {s['significant_FDR05']} | {s['higher_PD']} | {s['lower_PD']} | {s['nonconverged']} |")
    lines += ['', '## Paired comparisons', '']
    for name in ['subset_effect', 'cell_adjustment_effect']:
        s = metrics[name]
        lines.append(f"- {name}: effect correlation {s['log2FC_Pearson']:.4f}; {s['retained_significant_same_direction']}/{s['reference_significant']} reference discoveries retain significance and direction; median absolute effect shift {s['median_abs_log2FC_shift_reference_significant']:.4f} log2 units.")
    for name in ['shortlist_20', 'robust_1852']:
        s = metrics[name]
        lines.append(f"- {name}: {s['subset_reference_retained_primary_direction_FDR05']} retained in the subset reference; {s['cell_adjusted_retained_primary_direction_FDR05']} retained after cell adjustment, each compared with the original primary direction.")
    s = metrics['robust_15_hallmark']
    lines += [f"- Original 15 Hallmark pathways: {s['subset_reference_retained_primary_direction_FDR05']} retained in the subset reference and {s['cell_adjusted_retained_primary_direction_FDR05']} after adjustment.", '',
              'Full gene, shortlist, robust-gene, and Hallmark comparison TSVs accompany this report. BH and numerical GPU calculations were checked against CPU references; source hashes were verified unchanged.', '',
              'Effect shifts and significance retention are descriptive. Screening counts are imperfect baseline composition proxies; attenuation does not establish mediation, and retained associations do not establish within-cell mechanisms or disease specificity. Both models reuse the same participants. No external validation or classifier was performed.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))


def run():
    try:
        status('RUNNING_MODELS', pid=os.getpid())
        subprocess.run(['/usr/local/bin/Rscript', str(ROOT / 'run_models.R')], check=True, cwd=WORK.parent)
        status('FINALIZING', pid=os.getpid())
        finalize()
        hashes = json.loads((ROOT / 'input_sha256.json').read_text())
        assert all(sha(Path(path)) == value for path, value in hashes.items()), 'Source checksum changed'
        status('COMPLETE', report='RESULTS.md', summary='analysis_summary.json')
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        status('FAILED', error=error)
        raise


if __name__ == '__main__':
    if '--run' in sys.argv:
        run()
    else:
        checked = prepare()
        status('LAUNCHING')
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = sys.prefix
        env['PATH'] = str(Path(sys.executable).resolve().parent) + os.pathsep + env['PATH']
        env['OMP_NUM_THREADS'] = '1'
        env['OPENBLAS_NUM_THREADS'] = '1'
        (ROOT / 'matplotlib_cache').mkdir(exist_ok=True)
        env['MPLCONFIGDIR'] = str(ROOT / 'matplotlib_cache')
        with (ROOT / 'run.log').open('ab', buffering=0) as log:
            job = subprocess.Popen([sys.executable, '-u', str(ROOT / 'launch.py'), '--run'], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=WORK.parent)
        save_json('job.json', {'pid': job.pid, 'log': str(ROOT / 'run.log'), 'device': checked['device_name']})
        print(json.dumps({'launched_pid': job.pid, 'preflight': checked}, indent=2))
