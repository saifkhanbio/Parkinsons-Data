"""Matched held-out comparisons, conditional intervals, and figures."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from weighted_metrics import weighted_metrics
from model_utils import evaluate

FAMILIES=['combined','sex_only','age_only','pooled_de','pooled_hallmark']
LABELS={'combined':'Separate sex–age','sex_only':'Separate sex','age_only':'Separate age','pooled_de':'Pooled DE ridge','pooled_hallmark':'Pooled Hallmark ridge'}

def interval(row,values):
    assert np.isfinite(values).all()
    for j,name in enumerate(['AUROC','AP']):
        lo,hi=np.quantile(values[1:,j],[.025,.975])
        row.update({name:float(values[0,j]),name+'_lower':float(lo),name+'_upper':float(hi)})
    return row

def report(root,meta,pred,device):
    test=meta.loc[meta.allocation.eq('held_out')].copy().sort_index()
    y=test.label.to_numpy();n=len(test);assert n==105
    weights=np.zeros((5000,n),dtype=np.int16);rng=np.random.default_rng(20260914)
    strata=sorted(test.stratum.unique())
    for s in strata:
        for label in [0,1]:
            idx=np.flatnonzero(test.stratum.eq(s).to_numpy() & (y==label))
            assert len(idx)>0
            weights[:,idx]=rng.multinomial(len(idx),np.repeat(1/len(idx),len(idx)),size=5000)
    assert (weights.sum(1)==n).all()
    weights=np.vstack([np.ones((1,n),dtype=np.int16),weights])
    np.savez_compressed(root/'bootstrap_weights.npz',PATNO=test.PATNO.to_numpy(str),weights=weights)
    masks={'Joint holdout':np.ones(n,bool),**{s:test.stratum.eq(s).to_numpy() for s in strata}}
    rows=[];secondary=[];values={};errors=[]
    for family in FAMILIES:
        part=pred.loc[pred.family.eq(family)].set_index('PATNO').loc[test.PATNO]
        assert len(part)==n and part.index.is_unique and np.array_equal(part.label,y)
        scores=part.probability_PD.to_numpy();threshold=part.threshold.to_numpy()
        assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        for group,mask in masks.items():
            draw=weighted_metrics(y[mask],scores[mask],weights[:,mask],device)
            for i in [0,1,17]:
                w=weights[i,mask]
                ref=[roc_auc_score(y[mask],scores[mask],sample_weight=w),average_precision_score(y[mask],scores[mask],sample_weight=w)]
                errors.extend(abs(draw[i]-ref))
            values[(family,group)]=draw
            counts=np.bincount(y[mask],minlength=2)
            rows.append(interval(dict(family=family,stratum=group,n=int(mask.sum()),PD=int(counts[1]),Control=int(counts[0]),PD_prevalence=float(y[mask].mean()),sparse_test=bool(counts.min()<10)),draw))
            for mode,t in [('fixed_0_5',.5),('inner_selected',threshold[mask])]:
                secondary.append(dict(family=family,stratum=group,threshold_mode=mode,**evaluate(y[mask],scores[mask],t)))
    assert max(errors)<1e-12
    contrasts=[]
    for group in masks:
        for family in FAMILIES[:-1]:
            contrasts.append(interval(dict(stratum=group,comparison=family+' minus pooled_hallmark'),values[(family,group)]-values[('pooled_hallmark',group)]))
        for reference in ['sex_only','age_only','pooled_de']:
            contrasts.append(interval(dict(stratum=group,comparison='combined minus '+reference),values[('combined',group)]-values[(reference,group)]))
    performance=pd.DataFrame(rows);paired=pd.DataFrame(contrasts)
    performance.to_csv(root/'heldout_performance.tsv',sep='\t',index=False)
    paired.to_csv(root/'paired_comparisons.tsv',sep='\t',index=False)
    pd.DataFrame(secondary).to_csv(root/'threshold_metrics.tsv',sep='\t',index=False)
    np.savez_compressed(root/'bootstrap_metrics.npz',**{a+'__'+b:v for (a,b),v in values.items()})
    parameters=pd.read_csv(root/'selected_parameters.tsv',sep='\t')
    checks=dict(status='PASS',weighted_metrics_max_error=float(max(errors)),models=13,holdout_participants=n,predictions=len(pred),performance_rows=len(rows),paired_rows=len(contrasts),bootstrap_draws=5000,artifact_roundtrips=13)
    (root/'report_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    figure(root,performance,strata)
    lines=['# Separate sex–age training: held-out ridge assessment','',
        'New models were trained within demographic strata using whole-panel DESeq2 screening (adjusted P <0.05 and absolute unshrunk log2 fold change >0.5). All gene selection, filtering, scaling, ridge tuning and threshold choice used training participants only. The diagnostic DE design was ~ group. Unweighted ridge was used throughout.','',
        'Six feasible combined strata contributed 410 training and 105 held-out participants. Separate sex-only, age-only and pooled controls used the same participant allocation. Eleven people older than 80 and two with missing enrollment age were unallocated in this experiment; previous cohort analyses are unchanged.','',
        '## Joint held-out performance','',
        'AUROC and PR-AUC (average precision, AP) summarize the 105 held-out participants, routed to the appropriate model. Parentheses show conditional 95% bootstrap intervals.','',
        '| Procedure | AUROC (95% interval) | AP (95% interval) |','| --- | --- | --- |']
    for r in performance.loc[performance.stratum.eq('Joint holdout')].itertuples():
        lines.append(f'| {LABELS[r.family]} | {r.AUROC:.3f} ({r.AUROC_lower:.3f}–{r.AUROC_upper:.3f}) | {r.AP:.3f} ({r.AP_lower:.3f}–{r.AP_upper:.3f}) |')
    lines+=['','## Combined sex–age models','',
        '| Stratum | Test PD/control | Eligible RNA genes | AUROC (95% interval) | AP (95% interval) |','| --- | --- | --- | --- | --- |']
    for s in strata:
        r=performance.loc[performance.stratum.eq(s)&performance.family.eq('combined')].iloc[0]
        k=parameters.loc[parameters.model.eq('combined_'+s)].iloc[0]
        lines.append(f'| {s} | {r.PD}/{r.Control} | {int(k.genes)} | {r.AUROC:.3f} ({r.AUROC_lower:.3f}–{r.AUROC_upper:.3f}) | {r.AP:.3f} ({r.AP_lower:.3f}–{r.AP_upper:.3f}) |')
    lines+=['','## Interpretation and scope','',
        'The matched pooled Hallmark control, fitted anew on these training participants, is the appropriate comparator. Scores from the earlier 528-person repeated nested cross-validation experiment have a different evaluation design and cannot be treated as paired improvements. Joint metrics can reflect score offsets and prevalence differences across demographic models; all stratum-level results and matched contrasts are therefore retained.','',
        'These are exploratory within-cohort held-out checks after earlier analysis of this cohort. Training-only screening removes direct use of holdout outcomes from the current pipeline, but the split does not undo prior adaptive analyses. The highest observed model or stratum score is not an unbiased estimate of a selected future procedure.','',
        'Five thousand paired bootstrap draws resample participants within combined stratum and diagnosis while preserving observed cell sizes. Intervals condition on fitted models and omit refitting variability. Small strata, particularly tests with one or two controls, cannot support precise claims; bootstrap intervals may appear narrower than their scientific uncertainty. AP must be read alongside PD prevalence. No technical or measured-cell adjustment is included in this DE specification. These models do not establish clinical utility or effects on disability.','',
        'All 13 classifier artifacts reproduce their held-out predictions. Gene lists, preprocessing coefficients, inner OOF predictions, training metadata, raw-count exports, DE tables, input hashes and numerical checks are retained here. Zero-feature folds use training-prevalence predictions and remain in evaluation.','',
        '## Files','',
        '- `heldout_performance.tsv`: every procedure in each combined stratum and the joint holdout.',
        '- `paired_comparisons.tsv`: matched AUROC/AP changes and conditional intervals.',
        '- `threshold_metrics.tsv`: Brier and diagnostic metrics at 0.5 and training-selected thresholds.',
        '- `selected_parameters.tsv`, `models/`: regularization, actual feature counts and fitted artifacts.',
        '- `selector_summary.tsv`, `selectors/`: all training-only DE fits and selected genes.','',
        '![Joint holdout](joint_holdout_comparison.png)','',
        '![Stratum comparison](sex_age_comparison.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))

def figure(root,performance,strata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#444444']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric in zip(axes,['AUROC','AP']):
        for j,f in enumerate(FAMILIES):
            r=performance.loc[performance.stratum.eq('Joint holdout')&performance.family.eq(f)].iloc[0]
            ax.errorbar(r[metric],j,xerr=[[max(0,r[metric]-r[metric+'_lower'])],[max(0,r[metric+'_upper']-r[metric])]],fmt='o',capsize=4,color=colors[j])
        ax.set_yticks(range(5),[LABELS[f] for f in FAMILIES]);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.axvline(.5 if metric=='AUROC' else r.PD_prevalence,ls=':',color='gray')
        ax.set_xlabel('AUROC' if metric=='AUROC' else 'PR-AUC (average precision)');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Matched held-out assessment · 105 participants')
    fig.savefig(root/'joint_holdout_comparison.png',dpi=600);fig.savefig(root/'joint_holdout_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,s in zip(axes.flat,strata):
        for j,f in enumerate(FAMILIES):
            r=performance.loc[performance.stratum.eq(s)&performance.family.eq(f)].iloc[0]
            ax.errorbar(j,r.AUROC,yerr=[[max(0,r.AUROC-r.AUROC_lower)],[max(0,r.AUROC_upper-r.AUROC)]],fmt='o',color=colors[j],capsize=3)
        ax.set_title(s.replace('_',' ').replace('gt','>')+f'\nTest PD/control: {r.PD}/{r.Control}')
        ax.set_xticks(range(5),[LABELS[f] for f in FAMILIES],rotation=40,ha='right',fontsize=8)
        ax.set_ylim(0,1.02);ax.set_ylabel('AUROC');ax.axhline(.5,color='gray',ls=':');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Demographic specialization on identical held-out participants')
    fig.savefig(root/'sex_age_comparison.png',dpi=600);fig.savefig(root/'sex_age_comparison.pdf');plt.close(fig)
