"""Matched FDR-only versus strict-screen and Hallmark ridge assessment."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from weighted_metrics import weighted_metrics
from model_utils import evaluate

LABELS={'sex_fdr':'Sex-only: FDR only','sex_strict':'Sex-only: FDR + fold change','pooled_fdr':'Pooled: FDR only','pooled_strict':'Pooled: FDR + fold change','hallmark':'Pooled Hallmark ridge'}

def interval(row,v):
    assert v.shape==(5001,2) and np.isfinite(v).all()
    for j,name in enumerate(['AUROC','AP']):
        lo,hi=np.quantile(v[1:,j],[.025,.975]);row.update({name:float(v[0,j]),name+'_lower':float(lo),name+'_upper':float(hi)})
    return row

def report(root,meta,new,prior,device):
    test=meta.loc[meta.allocation.eq('held_out')].sort_index();y=test.label.to_numpy()
    boot=np.load(root/'bootstrap_weights.npz');weights=boot['weights']
    np.testing.assert_array_equal(boot['PATNO'],test.PATNO);assert weights.shape==(5001,105)
    groups={'Joint holdout':np.ones(len(test),bool),**{s:test.sex.eq(s).to_numpy() for s in ['Female','Male']},**{s:test.stratum.eq(s).to_numpy() for s in sorted(test.stratum.unique())}}
    procedures={'sex_fdr':new.loc[new.family.eq('sex_only')],'sex_strict':prior.loc[prior.family.eq('sex_only')],
                'pooled_fdr':new.loc[new.family.eq('pooled_de')],'pooled_strict':prior.loc[prior.family.eq('pooled_de')],'hallmark':prior.loc[prior.family.eq('pooled_hallmark')]}
    values={};rows=[];secondary=[];errors=[]
    for name,frame in procedures.items():
        part=frame.set_index('PATNO').loc[test.PATNO];assert len(part)==105 and part.index.is_unique and np.array_equal(part.label,y)
        p=part.probability_PD.to_numpy();threshold=part.threshold.to_numpy()
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        for group,mask in groups.items():
            v=weighted_metrics(y[mask],p[mask],weights[:,mask],device);values[(name,group)]=v
            for j in [0,1,17]:
                w=weights[j,mask];ref=[roc_auc_score(y[mask],p[mask],sample_weight=w),average_precision_score(y[mask],p[mask],sample_weight=w)]
                errors.extend(abs(v[j]-ref))
            counts=np.bincount(y[mask],minlength=2)
            rows.append(interval(dict(procedure=name,group=group,n=int(mask.sum()),PD=int(counts[1]),Control=int(counts[0]),PD_prevalence=float(y[mask].mean()),sparse_test=bool(counts.min()<10)),v))
            for mode,t in [('fixed_0_5',.5),('inner_selected',threshold[mask])]:secondary.append(dict(procedure=name,group=group,threshold_mode=mode,**evaluate(y[mask],p[mask],t)))
    assert max(errors)<1e-12
    contrasts=[]
    for group in groups:
        for a,b in [('sex_fdr','sex_strict'),('pooled_fdr','pooled_strict'),('sex_fdr','hallmark'),('pooled_fdr','hallmark')]:
            contrasts.append(interval(dict(group=group,comparison=a+' minus '+b),values[(a,group)]-values[(b,group)]))
    performance=pd.DataFrame(rows);paired=pd.DataFrame(contrasts)
    original=pd.read_csv(root/'prior_heldout_performance.tsv',sep='\t');reproduced=0
    columns=['AUROC','AUROC_lower','AUROC_upper','AP','AP_lower','AP_upper']
    for name,family in [('sex_strict','sex_only'),('pooled_strict','pooled_de'),('hallmark','pooled_hallmark')]:
        for r in original.loc[original.family.eq(family)].itertuples():
            current=performance.loc[performance.procedure.eq(name)&performance.group.eq(r.stratum)].iloc[0]
            np.testing.assert_allclose(current[columns].to_numpy(float),[getattr(r,k) for k in columns],atol=1e-12,rtol=0);reproduced+=1
    performance.to_csv(root/'heldout_performance.tsv',sep='\t',index=False);paired.to_csv(root/'paired_comparisons.tsv',sep='\t',index=False)
    pd.DataFrame(secondary).to_csv(root/'threshold_metrics.tsv',sep='\t',index=False)
    np.savez_compressed(root/'bootstrap_metrics.npz',**{'__'.join(k):v for k,v in values.items()})
    params=pd.read_csv(root/'selected_parameters.tsv',sep='\t')
    lines=['# Removing the fold-change screen: pooled and sex-only ridge','',
        'The only selection change is removal of the absolute log2 fold-change >0.5 cutoff. Training-specific DESeq2 padj <0.05, the original folds, preprocessing, unweighted ridge C grid and threshold rule are retained. No DESeq2 fits were repeated.','',
        '## Matched performance on 105 held-out participants','',
        'There are 72 PD participants and 33 controls (PD prevalence 0.686). PR-AUC is average precision (AP). Parentheses are conditional 95% intervals from the original 5,000 paired stratum-by-diagnosis bootstrap draws.','',
        '| Procedure | AUROC (95% interval) | AP (95% interval) |','| --- | --- | --- |']
    for r in performance.loc[performance.group.eq('Joint holdout')].itertuples():
        lines.append(f'| {LABELS[r.procedure]} | {r.AUROC:.3f} ({r.AUROC_lower:.3f}–{r.AUROC_upper:.3f}) | {r.AP:.3f} ({r.AP_lower:.3f}–{r.AP_upper:.3f}) |')
    lines+=['','## Paired changes','',
        '| Comparison | AUROC change (95% interval) | AP change (95% interval) |','| --- | --- | --- |']
    for r in paired.loc[paired.group.eq('Joint holdout')].itertuples():
        lines.append(f'| {r.comparison} | {r.AUROC:+.3f} ({r.AUROC_lower:+.3f}–{r.AUROC_upper:+.3f}) | {r.AP:+.3f} ({r.AP_lower:+.3f}–{r.AP_upper:+.3f}) |')
    lines+=['','## New evaluated models','',
        '| Model | Training / test N | Eligible genes | Selected C |','| --- | --- | --- | --- |']
    for r in params.itertuples():lines.append(f'| {r.model} | {r.training_n} / {r.test_n} | {r.genes} | {r.C:g} |')
    lines+=['','## Interpretation and limits','',
        'Judge the cutoff change by paired comparisons against the same strict-screen model; compare against the matched pooled Hallmark ridge to assess whether this alternative improves on the benchmark. Joint sex-model scores can reflect probability offsets and demographic prevalence, so retain the sex and combined-stratum results alongside joint metrics.','',
        'This is exploratory reuse of a previously examined within-cohort holdout. Conditional intervals omit refitting and adaptive model-selection variability; small strata provide limited information. No new split restores an untouched assessment after previous cohort-wide analyses. The earlier 528-person repeated-CV scores are not matched comparators. The reused diagnostic DE model has no technical or measured-cell adjustment. These results do not establish clinical utility or cell-independent expression.','',
        '## Reproducibility','',
        'The 33 reused DE fits have verified training membership and source hashes. Every original strict selected list is contained in its new FDR-only list. Four original ridge artifacts reproduce their held-out probabilities, and all three new artifacts pass inference roundtrips. Cutoff-boundary, empty-mask and held-out perturbation checks pass. Original comparator metrics and bootstrap intervals reproduce.','',
        '- `heldout_performance.tsv`, `paired_comparisons.tsv`: all joint and subgroup metrics and intervals.',
        '- `threshold_metrics.tsv`, `model_metrics.tsv`: Brier and threshold-dependent performance.',
        '- `selector_summary.tsv`, `selectors/`: selected genes and precise training membership.',
        '- `models/`: actual evaluated models, coefficients/scaling and inner OOF predictions.',
        '- `input_sha256.json`, `final_audit.json`, `output_sha256.json`: provenance and numerical validation.','',
        '![FDR-only comparison](fdr_only_comparison.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    (root/'report_validation.json').write_text(json.dumps(dict(status='PASS',maximum_metric_error=float(max(errors)),original_intervals_reproduced=reproduced,bootstrap_draws=5000,performance_rows=len(performance),paired_rows=len(paired)),indent=2)+'\n')
    figure(root,performance)

def figure(root,performance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,metric in zip(axes,['AUROC','AP']):
        for j,name in enumerate(LABELS):
            r=performance.loc[performance.procedure.eq(name)&performance.group.eq('Joint holdout')].iloc[0]
            ax.errorbar(r[metric],j,xerr=[[max(0,r[metric]-r[metric+'_lower'])],[max(0,r[metric+'_upper']-r[metric])]],fmt='o',capsize=4,color='#0072B2' if name.endswith('fdr') else '#555555')
        ax.set_yticks(range(5),list(LABELS.values()));ax.invert_yaxis();ax.set_xlim(0,1)
        ax.axvline(.5 if metric=='AUROC' else 72/105,color='gray',ls=':');ax.grid(axis='x',alpha=.15)
        ax.set_xlabel('AUROC' if metric=='AUROC' else 'PR-AUC (average precision)')
    fig.suptitle('Removing the fold-change screen · Same 105 held-out participants')
    fig.savefig(root/'fdr_only_comparison.png',dpi=600);fig.savefig(root/'fdr_only_comparison.pdf');plt.close(fig)
