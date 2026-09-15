"""Paired RF-minus-ridge comparisons using the original bootstrap draws."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from model_utils import evaluate
from weighted_metrics import weighted_metrics

FAMILIES=['combined','sex_only','age_only','pooled_de','pooled_hallmark']
LABELS={'combined':'Separate sex–age','sex_only':'Separate sex','age_only':'Separate age','pooled_de':'Pooled DE','pooled_hallmark':'Pooled Hallmark'}

def interval(row,v):
    assert v.shape==(5001,2) and np.isfinite(v).all()
    for j,key in enumerate(['AUROC','AP']):
        lo,hi=np.quantile(v[1:,j],[.025,.975]);row.update({key:float(v[0,j]),key+'_lower':float(lo),key+'_upper':float(hi)})
    return row

def report(root,meta,rf,ridge,device):
    test=meta.loc[meta.allocation.eq('held_out')].sort_index();y=test.label.to_numpy()
    boot=np.load(root/'bootstrap_weights.npz');weights=boot['weights'];np.testing.assert_array_equal(boot['PATNO'],test.PATNO)
    assert weights.shape==(5001,105) and (weights[0]==1).all()
    strata=sorted(test.stratum.unique())
    groups={'Joint holdout':np.ones(len(test),bool),**{s:test.stratum.eq(s).to_numpy() for s in strata}}
    groups.update({s:test.sex.eq(s).to_numpy() for s in ['Female','Male']})
    groups.update({'Age '+s:test.age_band.eq(s).to_numpy() for s in sorted(test.age_band.unique())})
    rows=[];secondary=[];values={};errors=[]
    for algorithm,frame in [('Ridge',ridge),('RF',rf)]:
        for family in FAMILIES:
            part=frame.loc[frame.family.eq(family)].set_index('PATNO').loc[test.PATNO]
            assert len(part)==105 and part.index.is_unique and np.array_equal(part.label,y)
            p=part.probability_PD.to_numpy();threshold=part.threshold.to_numpy();assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            for group,mask in groups.items():
                v=weighted_metrics(y[mask],p[mask],weights[:,mask],device);values[(algorithm,family,group)]=v
                for j in [0,1,17]:
                    w=weights[j,mask];ref=[roc_auc_score(y[mask],p[mask],sample_weight=w),average_precision_score(y[mask],p[mask],sample_weight=w)]
                    errors.extend(abs(v[j]-ref))
                counts=np.bincount(y[mask],minlength=2)
                rows.append(interval(dict(algorithm=algorithm,family=family,group=group,n=int(mask.sum()),PD=int(counts[1]),Control=int(counts[0]),PD_prevalence=float(y[mask].mean()),sparse_test=bool(counts.min()<10)),v))
                for mode,t in [('fixed_0_5',.5),('inner_selected',threshold[mask])]:secondary.append(dict(algorithm=algorithm,family=family,group=group,threshold_mode=mode,**evaluate(y[mask],p[mask],t)))
    assert max(errors)<1e-12
    comparisons=[]
    for group in groups:
        for family in FAMILIES:
            comparisons.append(interval(dict(group=group,family=family,comparison='RF minus matching ridge'),values[('RF',family,group)]-values[('Ridge',family,group)]))
            if family!='pooled_hallmark':
                comparisons.append(interval(dict(group=group,family=family,comparison='RF minus pooled Hallmark ridge'),values[('RF',family,group)]-values[('Ridge','pooled_hallmark',group)]))
    performance=pd.DataFrame(rows);paired=pd.DataFrame(comparisons)
    # Reproduce all previously reported ridge point estimates and bootstrap intervals.
    old=pd.read_csv(root/'ridge_heldout_performance.tsv',sep='\t')
    cols=['AUROC','AUROC_lower','AUROC_upper','AP','AP_lower','AP_upper']
    for r in old.itertuples():
        current=performance.loc[performance.algorithm.eq('Ridge')&performance.family.eq(r.family)&performance.group.eq(r.stratum)].iloc[0]
        np.testing.assert_allclose(current[cols].to_numpy(float),[getattr(r,k) for k in cols],atol=1e-12,rtol=0)
    performance.to_csv(root/'heldout_performance.tsv',sep='\t',index=False);paired.to_csv(root/'paired_comparisons.tsv',sep='\t',index=False)
    pd.DataFrame(secondary).to_csv(root/'threshold_metrics.tsv',sep='\t',index=False)
    np.savez_compressed(root/'bootstrap_metrics.npz',**{'__'.join(k):v for k,v in values.items()})
    params=pd.read_csv(root/'selected_parameters.tsv',sep='\t')
    report_rows=['# Random forest versus ridge on demographic strata','',
        'All 13 forests use the same training participants, held-out participants, inner folds and eligible gene sets as the corresponding ridge models. No DESeq2 fits were repeated. A four-configuration unweighted random-forest grid was selected within training folds.','',
        '## Joint performance on the same 105 held-out participants','',
        'PR-AUC is average precision (AP). The test set includes 72 PD participants and 33 controls (PD prevalence 0.686). Results pool predictions from the appropriate stratum-specific models.','',
        '| Training approach | Ridge AUROC | RF AUROC | Ridge AP | RF AP |','| --- | --- | --- | --- | --- |']
    for family in FAMILIES:
        a=performance.loc[performance.algorithm.eq('Ridge')&performance.family.eq(family)&performance.group.eq('Joint holdout')].iloc[0]
        b=performance.loc[performance.algorithm.eq('RF')&performance.family.eq(family)&performance.group.eq('Joint holdout')].iloc[0]
        report_rows.append(f'| {LABELS[family]} | {a.AUROC:.3f} | {b.AUROC:.3f} | {a.AP:.3f} | {b.AP:.3f} |')
    report_rows+=['','## Paired changes: RF minus matching ridge','',
        'Parentheses give conditional 95% bootstrap intervals using the exact 5,000 original stratum-by-diagnosis resamples.','',
        '| Training approach | AUROC change (95% interval) | AP change (95% interval) |','| --- | --- | --- |']
    for r in paired.loc[paired.group.eq('Joint holdout')&paired.comparison.eq('RF minus matching ridge')].itertuples():
        report_rows.append(f'| {LABELS[r.family]} | {r.AUROC:+.3f} ({r.AUROC_lower:+.3f}–{r.AUROC_upper:+.3f}) | {r.AP:+.3f} ({r.AP_lower:+.3f}–{r.AP_upper:+.3f}) |')
    report_rows+=['','## Combined sex–age models','',
        '| Stratum | Test PD/control | Genes | Ridge AUROC | RF AUROC | RF AP |','| --- | --- | --- | --- | --- | --- |']
    for group in strata:
        a=performance.loc[performance.algorithm.eq('Ridge')&performance.family.eq('combined')&performance.group.eq(group)].iloc[0]
        b=performance.loc[performance.algorithm.eq('RF')&performance.family.eq('combined')&performance.group.eq(group)].iloc[0]
        n=int(params.loc[params.model.eq('combined_'+group),'genes'].iloc[0])
        report_rows.append(f'| {group} | {b.PD}/{b.Control} | {n} | {a.AUROC:.3f} | {b.AUROC:.3f} | {b.AP:.3f} |')
    report_rows+=['','## Interpretation','',
        'Use paired differences and intervals to assess the algorithm change, rather than selecting the largest observed score. RF cannot recover molecular information when the unchanged DE screen selects no eligible genes; those models return the training prevalence. Their AP reflects test prevalence and their AUROC is 0.5.','',
        'The matched pooled Hallmark model is fitted on this experiment’s 410 training participants. The earlier 528-person repeated nested-CV results use different evaluation partitions and are not direct comparators. Joint scores from specialized models can reflect differences in demographic prevalence and probability offsets; the full subgroup tables retain all strata and both sex and age aggregations.','',
        'This is exploratory reuse of a holdout already inspected during model development. Conditional intervals omit refitting and adaptive model-selection variability; especially small diagnostic strata cannot support precise generalization claims. These results do not establish clinical utility or effects on disability. No technical or measured-cell adjustment is included in the reused DE specification. Impurity importances are descriptive and are not causal gene effects.','',
        '## Validation and saved outputs','',
        'All 123 training-specific DE masks were independently checked. Original held-out ridge predictions, eligible genes and scaling were verified in all 13 models. All RF artifact inference roundtrips passed. Original ridge performance and bootstrap intervals reproduce numerically. Held-out perturbation and empty-feature checks passed.','',
        '- `heldout_performance.tsv`: AUROC/AP and conditional intervals, joint and subgroup results.',
        '- `paired_comparisons.tsv`: RF minus matched ridge and pooled Hallmark ridge.',
        '- `threshold_metrics.tsv`: Brier, sensitivity, specificity and balanced accuracy.',
        '- `models/`: evaluated forests, actual features, preprocessing, importances and inner OOF probabilities.',
        '- `selected_parameters.tsv`, `inner_tuning_scores.tsv`: complete tuning evidence.',
        '- `input_sha256.json`, `final_audit.json`, `output_sha256.json`: provenance and validation.','',
        '![Joint RF versus ridge](rf_ridge_comparison.png)','',
        '![Combined sex–age assessment](rf_sex_age_comparison.png)','']
    (root/'RESULTS.md').write_text('\n'.join(report_rows))
    (root/'report_validation.json').write_text(json.dumps(dict(status='PASS',maximum_metric_error=float(max(errors)),ridge_intervals_reproduced=len(old),bootstrap_draws=5000,performance_rows=len(performance),paired_rows=len(paired)),indent=2)+'\n')
    figure(root,performance,strata)

def figure(root,performance,strata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,metric in zip(axes,['AUROC','AP']):
        for algorithm,offset,color in [('Ridge',-.12,'#555555'),('RF',.12,'#0072B2')]:
            for j,family in enumerate(FAMILIES):
                r=performance.loc[performance.algorithm.eq(algorithm)&performance.family.eq(family)&performance.group.eq('Joint holdout')].iloc[0]
                ax.errorbar(r[metric],j+offset,xerr=[[max(0,r[metric]-r[metric+'_lower'])],[max(0,r[metric+'_upper']-r[metric])]],fmt='o',capsize=3,color=color,label=algorithm if j==0 else None)
        ax.set_yticks(range(5),[LABELS[f] for f in FAMILIES]);ax.invert_yaxis();ax.set_xlim(0,1);ax.grid(axis='x',alpha=.15)
        ax.axvline(.5 if metric=='AUROC' else 72/105,color='gray',ls=':');ax.set_xlabel('AUROC' if metric=='AUROC' else 'PR-AUC (average precision)');ax.legend(loc='lower left',fontsize=8)
    fig.suptitle('Random forest versus ridge · Same 105 held-out participants')
    fig.savefig(root/'rf_ridge_comparison.png',dpi=600);fig.savefig(root/'rf_ridge_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,group in zip(axes.flat,strata):
        for j,(algorithm,color) in enumerate([('Ridge','#555555'),('RF','#0072B2')]):
            r=performance.loc[performance.algorithm.eq(algorithm)&performance.family.eq('combined')&performance.group.eq(group)].iloc[0]
            ax.errorbar(j,r.AUROC,yerr=[[max(0,r.AUROC-r.AUROC_lower)],[max(0,r.AUROC_upper-r.AUROC)]],fmt='o',capsize=4,color=color)
        ax.set_title(group.replace('_',' ').replace('gt','>')+f'\nTest PD/control: {r.PD}/{r.Control}')
        ax.set_xticks([0,1],['Ridge','RF']);ax.set_xlim(-.5,1.5);ax.set_ylim(0,1.03);ax.set_ylabel('AUROC');ax.axhline(.5,color='gray',ls=':')
    fig.suptitle('Separate sex–age models · Identical training-selected genes')
    fig.savefig(root/'rf_sex_age_comparison.png',dpi=600);fig.savefig(root/'rf_sex_age_comparison.pdf');plt.close(fig)
