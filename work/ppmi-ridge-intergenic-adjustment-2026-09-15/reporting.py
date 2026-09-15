"""Paired repeated nested-CV uncertainty for Hallmark and measured-cell models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from adjustment import FAMILIES,evaluate

PROTOCOLS=["participant_stratified","batch_grouped"]
BOOTSTRAPS=5000
SEED=20260914


def weighted_metrics(y,scores,weights,device):
    order=np.argsort(-scores,kind="stable")
    ends=np.r_[np.flatnonzero(np.diff(scores[order])!=0),len(y)-1]
    labels=torch.as_tensor(y[order],dtype=torch.float64,device=device)
    endpoints=torch.as_tensor(ends,dtype=torch.long,device=device)
    result=[]
    for start in range(0,len(weights),512):
        w=torch.as_tensor(weights[start:start+512,order],dtype=torch.float64,device=device)
        tp=torch.cumsum(w*labels,dim=1)[:,endpoints]
        fp=torch.cumsum(w*(1-labels),dim=1)[:,endpoints]
        dtp=torch.diff(tp,prepend=torch.zeros_like(tp[:,:1]),dim=1)
        dfp=torch.diff(fp,prepend=torch.zeros_like(fp[:,:1]),dim=1)
        positive,negative=tp[:,-1],fp[:,-1]
        auc=torch.sum(dfp*(tp-.5*dtp),dim=1)/(positive*negative)
        precision=torch.where(tp+fp>0,tp/(tp+fp),0.)
        ap=torch.sum(dtp*precision,dim=1)/positive
        v=torch.stack([auc,ap],dim=1)
        v[(positive==0)|(negative==0)]=float("nan")
        result.append(v.cpu().numpy())
    return np.concatenate(result)


def bootstrap_weights(meta,protocol,rng):
    if protocol=="participant_stratified":
        weights=np.zeros((BOOTSTRAPS,len(meta)),dtype=np.int16)
        for label in [0,1]:
            where=np.flatnonzero(meta.label.to_numpy()==label)
            weights[:,where]=rng.multinomial(len(where),np.full(len(where),1/len(where)),size=BOOTSTRAPS)
        assert np.all(weights.sum(1)==len(meta))
    else:
        levels,codes=np.unique(meta.batch.astype(str),return_inverse=True)
        w=rng.multinomial(len(levels),np.full(len(levels),1/len(levels)),size=BOOTSTRAPS)
        weights=w[:,codes].astype(np.int16)
        for i in range(len(levels)):assert np.all(weights[:,codes==i]==w[:,[i]])
    return np.vstack([np.ones((1,len(meta))),weights])


def add_interval(row,values):
    valid=np.isfinite(values[1:]).all(axis=1)
    row.update(bootstrap_valid=int(valid.sum()),bootstrap_invalid=int((~valid).sum()))
    assert valid.sum()>BOOTSTRAPS*.8,"Too many undefined bootstrap draws"
    for j,metric in enumerate(["AUROC","AP"]):
        row[f"{metric}_mean"]=float(values[0,j])
        low,high=np.quantile(values[1:][valid,j],[.025,.975])
        row[f"{metric}_ci_lower"]=float(low);row[f"{metric}_ci_upper"]=float(high)
    return row

def report(root,meta,pred,metrics,device):
    def table(name,x):pd.DataFrame(x).to_csv(root/name,sep='\t',index=False)
    y=meta.label.to_numpy();primary=[];differences=[];secondary=[];archive={};loss_summary=[]
    # Independent tie-aware metric validation.
    yy=np.array([0,1,0,1,0,1]);scores=np.array([.1,.1,.5,.5,.9,.9]);w0=np.random.default_rng(5).integers(1,5,(20,6))
    expected=np.array([[roc_auc_score(yy,scores,sample_weight=w),average_precision_score(yy,scores,sample_weight=w)] for w in w0]);np.testing.assert_allclose(weighted_metrics(yy,scores,w0,device),expected,atol=1e-12,rtol=0)
    repeats=metrics.groupby(['protocol','family','repeat'],as_index=False)[['AUROC','AP','train_AUROC','train_AP']].mean();table('primary_per_repeat.tsv',repeats)
    for pi,proto in enumerate(PROTOCOLS):
        weights=bootstrap_weights(meta,proto,np.random.default_rng(SEED+1000*pi));model_draws={};losses={}
        for family in FAMILIES:
            draws=[];brier=[];logs=[]
            for repeat in [1,2,3]:
                p=pred[(pred.protocol==proto)&(pred.family==family)&(pred.repeat==repeat)].set_index('PATNO').loc[meta.PATNO];assert len(p)==528 and np.array_equal(p.label,y)
                prob=p.probability_PD.to_numpy();fd=[]
                for fold in [1,2,3,4,5]:
                    take=p.fold.to_numpy()==fold;v=weighted_metrics(y[take],prob[take],weights[:,take],device)
                    row=metrics[(metrics.protocol==proto)&(metrics.family==family)&(metrics.repeat==repeat)&(metrics.fold==fold)].iloc[0]
                    np.testing.assert_allclose(v[0],[row.AUROC,row.AP],atol=1e-12,rtol=0);fd.append(v)
                draws.append(np.mean(fd,axis=0));brier.append((prob-y)**2);logs.append(-(y*np.log(np.clip(prob,1e-12,1))+(1-y)*np.log(np.clip(1-prob,1e-12,1))))
                for mode,t in [('fixed_0_5',.5),('inner_selected',p.threshold.to_numpy())]:secondary.append(dict(protocol=proto,family=family,repeat=repeat,threshold_mode=mode,**evaluate(y,prob,t)))
            values=np.mean(draws,axis=0);model_draws[family]=values;archive[proto+'__'+family]=values
            row=add_interval(dict(protocol=proto,family=family),values);r=repeats[(repeats.protocol==proto)&(repeats.family==family)]
            row.update(train_AUROC=float(r.train_AUROC.mean()),train_AP=float(r.train_AP.mean()),AUROC_repeat_min=float(r.AUROC.min()),AUROC_repeat_max=float(r.AUROC.max()))
            primary.append(row)
            for name,b in [('Brier',brier),('log_loss',logs)]:
                v=weights@np.mean(b,axis=0)/weights.sum(1);losses[(family,name)]=v
                lo,hi=np.quantile(v[1:],[.025,.975]);loss_summary.append(dict(protocol=proto,family=family,metric=name,estimate=v[0],ci_lower=lo,ci_upper=hi))
        differences.append(add_interval(dict(protocol=proto,comparison='intergenic_adjusted minus rna'),model_draws['intergenic_adjusted']-model_draws['rna']))
        for name in ['Brier','log_loss']:
            v=losses[('intergenic_adjusted',name)]-losses[('rna',name)];lo,hi=np.quantile(v[1:],[.025,.975]);loss_summary.append(dict(protocol=proto,family='adjusted_minus_rna',metric=name,estimate=v[0],ci_lower=lo,ci_upper=hi))
    primary=pd.DataFrame(primary);differences=pd.DataFrame(differences)
    table('primary_summary.tsv',primary);table('paired_primary_comparisons.tsv',differences);table('secondary_pooled_metrics.tsv',secondary);table('probability_loss_summary.tsv',loss_summary)
    np.savez_compressed(root/'bootstrap_metrics.npz',metric_names=np.array(['AUROC','AP']),**archive)
    a=metrics[metrics.family.eq('intergenic_adjusted')].set_index(['protocol','repeat','fold']);b=metrics[metrics.family.eq('rna')].set_index(['protocol','repeat','fold'])
    table('paired_fold_differences.tsv',(a[['AUROC','AP']]-b[['AUROC','AP']]).reset_index())
    lines=['# Training-only intergenic-read adjustment comparison','','Same 528 participants, Hallmark candidate genes, nested folds and ridge C grid. The adjusted procedure removes each gene’s training-estimated linear association with intergenic-read percentage before residual standardization and ridge fitting. Previous models and calibrated results remain unchanged.','','## Held-out discrimination','','| Protocol | Model | AUROC (95% interval) | AP (95% interval) |','|---|---|---|---|']
    for r in primary.itertuples():lines.append(f'| {r.protocol} | {r.family} | {r.AUROC_mean:.3f} ({r.AUROC_ci_lower:.3f} to {r.AUROC_ci_upper:.3f}) | {r.AP_mean:.3f} ({r.AP_ci_lower:.3f} to {r.AP_ci_upper:.3f}) |')
    lines+=['','Primary metrics are mean within outer folds, averaged over repeats. Intervals use 5,000 paired fixed-prediction bootstrap draws.','','| Protocol | AUROC change (95% interval) | AP change (95% interval) |','|---|---|---|']
    for r in differences.itertuples():lines.append(f'| {r.protocol} | {r.AUROC_mean:+.3f} ({r.AUROC_ci_lower:+.3f} to {r.AUROC_ci_upper:+.3f}) | {r.AP_mean:+.3f} ({r.AP_ci_lower:+.3f} to {r.AP_ci_upper:+.3f}) |')
    lines+=['','## Scope and interpretation','','This single adjustment was motivated by the earlier error audit. The nuisance regression uses no diagnosis, age, sex or cell-count information; therefore it may remove disease-associated expression correlated with intergenic reads. It is not a causal technical-effect estimate. The adjusted inference pipeline requires the same intergenic percentage measurement for each new sample. It does not correct unknown batches or other technical variables.','','Unchanged gene filtering and original RNA-model prediction/tuning reproduction isolate the preprocessing comparison. Residual scaling and ridge C selection remain training-specific. Secondary pooled OOF metrics and Brier/log-loss comparisons are saved separately; they do not replace the primary within-fold AUROC/AP estimands. This experiment does not apply or alter the previously retained calibrators.','','Bootstrap uncertainty resamples participants within diagnosis or whole phase–plate batches, shares draws across models/repeats, and omits retraining and adaptive-search variation. All results are exploratory on a previously studied cohort. No new exclusions or outcome-driven technical thresholds were introduced.','','## Reproducibility','','Saved artifacts include per-fold models and nuisance slopes, exact split plans, cached inner predictions, selected parameters, extrapolation flags, final five-fold tuning and two full-cohort development models. Final development fits add no new performance evidence. Use predict.py with complete raw counts and, for the adjusted model, intergenic_percent metadata. Source integrity, benchmark agreement, held-out perturbations, OLS/CPU/GPU agreement and inference are checked.','','![Comparison](intergenic_adjustment_comparison.png)','','Preprocessing reference: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage','']
    (root/'RESULTS.md').write_text('\n'.join(lines));figure(root,primary,repeats)
    return dict(primary_results=primary.to_dict('records'),paired_results=differences.to_dict('records'))

def figure(root,primary,repeats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,9));colors=['#64748B','#C07126']
    for col,proto in enumerate(PROTOCOLS):
        for row,metric in enumerate(['AUROC','AP']):
            ax=axes[row,col];s=primary[primary.protocol==proto].set_index('family')
            for i,f in enumerate(FAMILIES):
                r=s.loc[f];point=r[metric+'_mean'];lo=r[metric+'_ci_lower'];hi=r[metric+'_ci_upper']
                ax.vlines(i,lo,hi,color=colors[i],lw=2);ax.scatter(i,point,color=colors[i],s=55);ax.text(i,hi+.009,f'{point:.3f}',ha='center')
                rr=repeats[(repeats.protocol==proto)&(repeats.family==f)][metric];ax.scatter(i+np.array([-.06,0,.06]),rr,color=colors[i],s=15,alpha=.5)
            ax.set_xticks([0,1],['Original ridge','Intergenic-adjusted\nridge']);ax.set_xlim(-.4,1.4)
            chance=.5 if metric=='AUROC' else 358/528;ax.axhline(chance,color='gray',ls='--');ax.set_ylim(max(0,min(chance,s[metric+'_ci_lower'].min())-.04),min(1,s[metric+'_ci_upper'].max()+.05))
            ax.set_ylabel('Mean within-fold '+metric);ax.grid(axis='y',alpha=.15)
            if row==0:ax.set_title(proto.replace('_',' ').capitalize(),pad=16)
            ax.text(-.13,1.05,'ABCD'[row*2+col],transform=ax.transAxes,fontweight='bold',fontsize=14)
    fig.suptitle('Training-only intergenic adjustment of Hallmark ridge',fontsize=14,y=.98)
    fig.subplots_adjust(left=.10,right=.96,top=.88,bottom=.15,wspace=.3,hspace=.37)
    fig.text(.10,.035,'528 participants • Same nested folds • Three repeats\nLarge points: mean performance; small points: repeat estimates; bars: conditional 95% intervals.\nAP: average precision. Dashed lines: AUROC 0.5 or cohort PD prevalence.',fontsize=9,linespacing=1.5)
    for ext in ['png','pdf','svg']:fig.savefig(root/('intergenic_adjustment_comparison.'+ext),dpi=600,facecolor='white')
    fig.savefig(root/'intergenic_adjustment_comparison_preview.png',dpi=110,facecolor='white');plt.close(fig)
