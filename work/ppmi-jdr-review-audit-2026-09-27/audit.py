"""Aggregate reviewer evidence from preserved outputs; no RNA model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SOURCES = {}


def read(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, sep='\t', dtype={'PATNO': str})


def save(name, data):
    pd.DataFrame(data).to_csv(ROOT/name, sep='\t', index=False)


def main():
    primary = read(WORK/'ppmi-deseq2-2026-09-12/primary_colData.tsv')
    classifier = read(WORK/'ppmi-classifier-2026-09-12/metadata.tsv')
    rows = []
    for name, frame in [('primary_558', primary), ('classifier_528', classifier)]:
        assert frame.PATNO.is_unique
        counts = pd.crosstab(frame.batch, frame.group)[['Control', 'PD']]
        counts['n'] = counts.sum(axis=1)
        counts['PD_fraction'] = counts.PD/counts.n
        counts['mixed'] = (counts.Control > 0) & (counts.PD > 0)
        save(f'{name}_batch_counts.tsv', counts.reset_index())
        y = frame.group.eq('PD').to_numpy(float)
        x = pd.get_dummies(frame.batch, dtype=float).to_numpy()
        residual = y-x @ np.linalg.lstsq(x, y, rcond=None)[0]
        r2 = 1-np.sum(residual**2)/np.sum((y-y.mean())**2)
        rows.append(dict(cohort=name, n=len(frame), batches=len(counts), mixed_batches=int(counts.mixed.sum()),
                         single_diagnosis_batches=int((~counts.mixed).sum()),
                         people_in_mixed_batches=int(counts.loc[counts.mixed, 'n'].sum()),
                         people_in_single_diagnosis_batches=int(counts.loc[~counts.mixed, 'n'].sum()),
                         min_batch_size=int(counts.n.min()), max_batch_size=int(counts.n.max()),
                         diagnosis_batch_R2=float(r2), diagnosis_batch_VIF=float(1/(1-r2))))
    save('batch_balance_summary.tsv', rows)
    plan_path = WORK/'ppmi-mapped-gene-classifier-2026-09-12/fold_plan.json'
    SOURCES[str(plan_path)] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = json.loads(plan_path.read_text())
    y = classifier.label.to_numpy()
    b = classifier.batch.to_numpy(str)
    rows = []
    for s in plan:
        train, test = np.array(s['train_indices']), np.array(s['test_indices'])
        assert not np.intersect1d(train, test).size
        rates = {name:float((y[train[b[train] == name]].sum()+1)/(np.sum(b[train] == name)+2)) for name in np.unique(b[train])}
        probability = np.array([rates.get(name, float(y[train].mean())) for name in b[test]])
        if s['protocol'] == 'batch_grouped':
            assert not set(b[train]) & set(b[test]) and np.ptp(probability) == 0
        rows.append(dict(**{k:s[k] for k in ['repeat', 'protocol', 'fold']}, n_test=len(test),
                         unseen_batch_samples=int(sum(name not in rates for name in b[test])),
                         AUROC=float(roc_auc_score(y[test], probability))))
    save('batch_only_fold_metrics.tsv', rows)
    repeats = pd.DataFrame(rows).groupby(['repeat', 'protocol']).AUROC.mean().reset_index()
    save('batch_only_repeat_metrics.tsv', repeats)
    save('batch_only_summary.tsv', repeats.groupby('protocol').AUROC.agg(['mean', 'min', 'max']).reset_index())

    a = read(WORK/'ppmi-deseq2-2026-09-12/primary/results.tsv').set_index('Geneid')
    b = read(WORK/'ppmi-results-review-2026-09-12/site_race_results.tsv').set_index('Geneid').loc[a.index]
    assert a.index.is_unique and len(a) == len(b) == 21888
    both = a.pvalue.notna() & b.pvalue.notna()
    sa, sb = a.padj.lt(.05), b.padj.lt(.05)
    same = np.sign(a.log2FoldChange) == np.sign(b.log2FoldChange)
    rows = []
    for label, mask in [('all_jointly_testable', both), ('primary_significant', both & sa),
                         ('gained_significance', both & ~sa & sb), ('lost_significance', both & sa & ~sb)]:
        rows.append(dict(subset=label, genes=int(mask.sum()),
                         effect_correlation=float(a.loc[mask, 'log2FoldChange'].corr(b.loc[mask, 'log2FoldChange'])),
                         median_absolute_effect_shift=float((b.loc[mask, 'log2FoldChange']-a.loc[mask, 'log2FoldChange']).abs().median()),
                         median_SE_ratio_site_primary=float((b.loc[mask, 'lfcSE']/a.loc[mask, 'lfcSE']).median()),
                         same_direction_fraction=float(same[mask].mean())))
    save('site_race_effect_uncertainty.tsv', rows)
    discovered = []
    for name, d in [('primary', a), ('site_race', b)]:
        p = d.pvalue.dropna()
        no_filter = multipletests(p.to_numpy(), method='fdr_bh')[1]
        discovered.append(dict(model=name, tested_p_values=len(p), adjusted_p_values=int(d.padj.notna().sum()),
                               FDR05=int(d.padj.lt(.05).sum()), FDR05_without_independent_filter=int((no_filter < .05).sum())))
    save('site_race_FDR_comparison.tsv', discovered)
    transition = dict(primary_significant=int(sa.sum()), site_race_significant=int(sb.sum()),
                      shared_significant=int((sa & sb).sum()), shared_same_direction=int((sa & sb & same).sum()),
                      gained=int((~sa & sb).sum()), lost=int((sa & ~sb).sum()))
    (ROOT/'site_race_transitions.json').write_text(json.dumps(transition, indent=2)+'\n')
    stability = read(WORK/'ppmi-classifier-feature-stability-2026-09-12/annotated_stability.tsv')
    stability = stability[stability.penalty.eq('elasticnet')]
    setting_count = len(stability[['protocol', 'family']].drop_duplicates())
    stable_by_gene = stability.groupby('reference_gene_name').stable.agg(['all', 'size'])
    retained = stable_by_gene.index[stable_by_gene['all'] & stable_by_gene['size'].eq(setting_count)]
    stability = stability[stability.reference_gene_name.isin(retained)]
    assert not stability.empty and stability.stable.all()
    assert (stability[['repeat_1_retained', 'repeat_2_retained', 'repeat_3_retained']].min(axis=1) >= 4).all()
    assert (stability.sign_consistency >= .9).all()
    cols = ['reference_gene_name', 'protocol', 'family', 'retained_fits', 'repeat_1_retained', 'repeat_2_retained',
            'repeat_3_retained', 'positive_fits', 'negative_fits', 'sign_consistency']
    save('stable_gene_verification.tsv', stability[cols])
    cbc = read(WORK/'ppmi-blood-cell-adjustment-2026-09-12/blood_covariates.tsv')
    counts = pd.crosstab(cbc.month_gap_lab_minus_rna, cbc.group).reset_index()
    save('CBC_month_counts.tsv', counts)
    (ROOT/'audit_input_sha256.json').write_text(json.dumps(SOURCES, indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(transition))
    print((ROOT/'batch_balance_summary.tsv').read_text())
    print((ROOT/'batch_only_summary.tsv').read_text())
    print((ROOT/'site_race_FDR_comparison.tsv').read_text())


if __name__ == '__main__':
    main()
