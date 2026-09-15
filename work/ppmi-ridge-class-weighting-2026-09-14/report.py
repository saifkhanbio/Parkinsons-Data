"""Paired discrimination and descriptive calibration/operating-point assessment."""
import sys,json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from weighting import STEP2,ARMS,evaluate
sys.path.insert(0,str(STEP2))
from reporting import weighted_metrics,bootstrap_weights,add_interval,BOOTSTRAPS,SEED,PROTOCOLS

LABELS={'unweighted':'Unweighted ridge','balanced':'Balanced ridge','selected_weight':'Inner-selected weighting'}

def report(root,meta,pred,folds,params,device):
    def table(name,x):pd.DataFrame(x).to_csv(root/name,sep='\t',index=False)
    y=meta.label.to_numpy();primary=[];contrasts=[];archive={};errors=[]
    repeats=folds.groupby(['protocol','arm','repeat'],as_index=False)[['AUROC','AP','Brier','log_loss','train_AUROC','train_AP']].mean()
    table('primary_per_repeat.tsv',repeats)
    for pi,protocol in enumerate(PROTOCOLS):
        weights=bootstrap_weights(meta,protocol,np.random.default_rng(SEED+1000*pi));draws={}
        for arm in ARMS:
            repeated=[]
            for repeat in [1,2,3]:
                p=pred[(pred.protocol==protocol)&(pred.arm==arm)&(pred['repeat']==repeat)].set_index('PATNO').loc[meta.PATNO]
                assert len(p)==528 and np.array_equal(p.label,y)
                scores=p.probability_PD.to_numpy();dd=[]
                for fold in [1,2,3,4,5]:
                    mask=p.fold.to_numpy()==fold
                    values=weighted_metrics(y[mask],scores[mask],weights[:,mask],device)
                    actual=folds[(folds.protocol==protocol)&(folds.arm==arm)&(folds['repeat']==repeat)&(folds.fold==fold)].iloc[0]
                    np.testing.assert_allclose(values[0],[actual.AUROC,actual.AP],atol=1e-12,rtol=0)
                    for sample in [1,17]:
                        if np.isfinite(values[sample]).all():
                            expected=[roc_auc_score(y[mask],scores[mask],sample_weight=weights[sample,mask]),average_precision_score(y[mask],scores[mask],sample_weight=weights[sample,mask])]
                            errors.extend(abs(values[sample]-expected))
                    dd.append(values)
                repeated.append(np.mean(dd,axis=0))
            values=np.mean(repeated,axis=0);draws[arm]=values;archive[f'{protocol}__{arm}']=values
            row=add_interval(dict(protocol=protocol,arm=arm),values)
            rr=repeats[(repeats.protocol==protocol)&(repeats.arm==arm)]
            for metric in ['AUROC','AP']:
                row[f'{metric}_repeat_min']=float(rr[metric].min());row[f'{metric}_repeat_max']=float(rr[metric].max())
            row.update(Brier_mean=float(rr.Brier.mean()),log_loss_mean=float(rr.log_loss.mean()))
            primary.append(row)
        for arm in ['selected_weight','balanced']:
            contrasts.append(add_interval(dict(protocol=protocol,comparison=f'{arm} minus unweighted'),draws[arm]-draws['unweighted']))
        print(f'Bootstrap complete: {protocol}',flush=True)
    assert max(errors)<1e-12
    primary=pd.DataFrame(primary);contrasts=pd.DataFrame(contrasts)
    table('primary_summary.tsv',primary);table('paired_primary_comparisons.tsv',contrasts)
    np.savez_compressed(root/'bootstrap_metrics.npz',metric_names=np.array(['AUROC','AP']),**archive)
    differences=[]
    for arm in ['selected_weight','balanced']:
        a=folds[folds.arm==arm].set_index(['protocol','repeat','fold'])
        b=folds[folds.arm=='unweighted'].set_index(['protocol','repeat','fold'])
        d=(a[['AUROC','AP','Brier','log_loss']]-b[['AUROC','AP','Brier','log_loss']]).reset_index()
        d['comparison']=f'{arm} minus unweighted';differences.append(d)
    differences=pd.concat(differences);table('paired_fold_differences.tsv',differences)
    table('paired_repeat_differences.tsv',differences.groupby(['protocol','comparison','repeat'],as_index=False)[['AUROC','AP','Brier','log_loss']].mean())
    selection=params.groupby(['protocol','arm','weight','C']).size().reset_index(name='outer_fold_count');table('weight_selection_counts.tsv',selection)
    secondary=[];calibration=[];bins=[]
    for (protocol,arm,repeat),part in pred.groupby(['protocol','arm','repeat']):
        yy=part.label.to_numpy();p=part.probability_PD.to_numpy()
        for mode,t in [('fixed_0_5',.5),('inner_selected',part.threshold.to_numpy())]:
            secondary.append(dict(protocol=protocol,arm=arm,repeat=repeat,threshold_mode=mode,**evaluate(yy,p,t)))
        z=logit(np.clip(p,1e-12,1-1e-12))
        cal=LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,tol=1e-9).fit(z[:,None],yy)
        assert cal.n_iter_[0]<10000
        intercept=brentq(lambda a:expit(z+a).mean()-yy.mean(),-100,100)
        calibration.append(dict(protocol=protocol,arm=arm,repeat=repeat,n=len(yy),observed_prevalence=yy.mean(),
           predicted_prevalence=p.mean(),Brier=np.mean((p-yy)**2),log_loss=log_loss(yy,p),
           calibration_in_the_large_intercept=intercept,calibration_intercept=float(cal.intercept_[0]),calibration_slope=float(cal.coef_[0,0])))
        ids=np.minimum((p*10).astype(int),9)
        for b in range(10):
            mask=ids==b
            if mask.any():bins.append(dict(protocol=protocol,arm=arm,repeat=repeat,bin=b,lower=b/10,upper=(b+1)/10,n=int(mask.sum()),mean_probability=float(p[mask].mean()),observed_fraction=float(yy[mask].mean())))
    secondary=pd.DataFrame(secondary);calibration=pd.DataFrame(calibration)
    table('secondary_pooled_OOF_metrics.tsv',secondary);table('calibration_per_repeat.tsv',calibration);table('calibration_bins.tsv',bins)
    secondary_summary=secondary.groupby(['protocol','arm','threshold_mode'],as_index=False)[['sensitivity','specificity','balanced_accuracy','accuracy','Brier','AUROC','average_precision']].mean()
    table('operating_point_summary.tsv',secondary_summary)
    table('calibration_summary.tsv',calibration.groupby(['protocol','arm'],as_index=False).mean(numeric_only=True).drop(columns=['repeat']))
    plot(root,primary,repeats,pd.DataFrame(bins))
    lines=['# Step 3: RNA-only ridge class weighting','','Same 528 participants, fixed Hallmark member universe, and original nested folds. PD is the positive class. Only training class weighting is varied.','','## Discrimination','','AUROC and AP are mean within-fold metrics averaged across three repeats. AP denotes average precision. Parentheses contain conditional 95% paired-bootstrap intervals.','','| Protocol | Procedure | AUROC (95% interval) | AP (95% interval) |','| --- | --- | --- | --- |']
    for r in primary.itertuples(index=False):lines.append(f'| {r.protocol} | {LABELS[r.arm]} | {r.AUROC_mean:.4f} ({r.AUROC_ci_lower:.4f}–{r.AUROC_ci_upper:.4f}) | {r.AP_mean:.4f} ({r.AP_ci_lower:.4f}–{r.AP_ci_upper:.4f}) |')
    lines += ['','## Paired changes versus unweighted','','| Protocol | Procedure | AUROC difference (95% interval) | AP difference (95% interval) |','| --- | --- | --- | --- |']
    for r in contrasts.itertuples(index=False):lines.append(f'| {r.protocol} | {r.comparison} | {r.AUROC_mean:+.4f} ({r.AUROC_ci_lower:+.4f}–{r.AUROC_ci_upper:+.4f}) | {r.AP_mean:+.4f} ({r.AP_ci_lower:+.4f}–{r.AP_ci_upper:+.4f}) |')
    lines += ['','## Operating points','','Mean repeat-specific pooled OOF sensitivity/specificity; thresholds are either fixed at 0.5 or selected inside training. These are descriptive summaries.','','| Protocol | Procedure | Threshold | Sensitivity | Specificity | Balanced accuracy |','| --- | --- | --- | --- | --- | --- |']
    for r in secondary_summary.itertuples(index=False):lines.append(f'| {r.protocol} | {LABELS[r.arm]} | {r.threshold_mode} | {r.sensitivity:.3f} | {r.specificity:.3f} | {r.balanced_accuracy:.3f} |')
    lines += ['','## Calibration and validation','','Brier/log loss and descriptive calibration intercept/slope are saved in calibration_per_repeat.tsv and calibration_summary.tsv. Lower Brier/log loss is better. Calibration-in-the-large fits an intercept with prediction log-odds as an offset; the separate intercept/slope model estimates both parameters. Ideal values are zero intercept and slope one. All calibration fits use saved OOF predictions for assessment only and are never applied to predictions. Reliability-bin observations are repeated participants, not independent samples.','','Every original unweighted fold reproduced its tuning, eligible genes, threshold and probabilities. Class weights are recorded for every inner partition and outer refit. Both inner and outer batch partitions remain disjoint. The selected-weight arm reuses the matching unweighted/balanced fitted model. Three full-data development artifacts are saved without a new performance claim.','',f'There were {BOOTSTRAPS:,} paired conditional bootstrap draws per protocol. Minimum valid primary draws: {int(primary.bootstrap_valid.min()):,}. Undefined draws are counted. Intervals omit model retraining and the broader adaptive sequence of comparisons; cohort calibration does not establish population risk calibration.','','![Discrimination](class_weight_comparison.png)','','![Calibration](calibration_comparison.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    validation=dict(weighted_metric_max_error=float(max(errors)),minimum_valid_primary_draws=int(primary.bootstrap_valid.min()),checks='PASS')
    (root/'report_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    return dict(primary_results=primary.to_dict(orient='records'),paired_results=contrasts.to_dict(orient='records'),report_validation=validation)

def plot(root,primary,repeats,bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#247BA0','#C06A27','#1B8A6B']
    fig,axes=plt.subplots(2,2,figsize=(11.5,8))
    for j,protocol in enumerate(PROTOCOLS):
        s=primary[primary.protocol==protocol].set_index('arm')
        for row,metric in enumerate(['AUROC','AP']):
            ax=axes[row,j]
            for i,arm in enumerate(ARMS):
                v=s.loc[arm];point=v[f'{metric}_mean'];low=v[f'{metric}_ci_lower'];high=v[f'{metric}_ci_upper']
                ax.vlines(i,low,high,color=colors[i],lw=2);ax.scatter(i,point,color=colors[i],s=55,zorder=3)
                rr=repeats[(repeats.protocol==protocol)&(repeats.arm==arm)].sort_values('repeat')[metric]
                ax.scatter(i+np.array([-.08,0,.08]),rr,color=colors[i],s=16,alpha=.4)
                ax.annotate(f'{point:.4f}',(i,high),xytext=(0,7),textcoords='offset points',ha='center')
            ax.set_xticks(range(3),['Unweighted','Balanced','Inner-selected\nweighting']);ax.set_xlim(-.4,2.4)
            ax.set_ylim(s[f'{metric}_ci_lower'].min()-.025,s[f'{metric}_ci_upper'].max()+.035)
            ax.set_ylabel('Mean within-fold '+('AUROC' if metric=='AUROC' else 'average precision'));ax.grid(axis='y',alpha=.15)
            ax.text(-.11,1.05,'ABCD'[row*2+j],transform=ax.transAxes,fontweight='bold',fontsize=13)
            if row==0:ax.set_title('Participant-stratified folds' if j==0 else 'Batch-grouped folds',pad=15)
    fig.suptitle('Class weighting in the RNA-only Hallmark ridge classifier',fontsize=15,y=.97)
    fig.text(.5,.92,'528 participants • Same genes and nested folds • Three repeats',ha='center')
    fig.subplots_adjust(left=.10,right=.975,top=.83,bottom=.17,hspace=.43,wspace=.28)
    fig.text(.1,.05,'Large points: mean over repeats. Small points: repeat estimates. Bars: conditional 95% bootstrap intervals.\nInner-selected weighting jointly tunes class weighting and regularization within training folds.',fontsize=9,linespacing=1.6)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'class_weight_comparison.{ext}',dpi=600,facecolor='white')
    fig.savefig(root/'class_weight_comparison_preview.png',dpi=120,facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.5))
    for j,protocol in enumerate(PROTOCOLS):
        ax=axes[j];ax.plot([0,1],[0,1],'--',color='#999999',lw=1)
        for i,arm in enumerate(ARMS):
            b=bins[(bins.protocol==protocol)&(bins.arm==arm)]
            # Pool bin counts across repeats descriptively; do not treat as independent n.
            grouped=b.groupby('bin').apply(lambda x:pd.Series({'p':np.average(x.mean_probability,weights=x.n),'y':np.average(x.observed_fraction,weights=x.n)}),include_groups=False)
            ax.plot(grouped.p,grouped.y,'o-',color=colors[i],label=LABELS[arm],ms=4)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted PD probability',ylabel='Observed PD fraction',title='Participant-stratified' if j==0 else 'Batch-grouped')
        ax.grid(alpha=.15)
    axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Descriptive calibration of held-out predictions',fontsize=15)
    fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.22,wspace=.25)
    fig.text(.08,.06,'Fixed-width probability bins pooled across three repeats for display. Participants recur across repeats.\nDashed line: perfect calibration. No recalibration or prevalence correction was applied.',fontsize=9,linespacing=1.6)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'calibration_comparison.{ext}',dpi=600,facecolor='white')
    fig.savefig(root/'calibration_comparison_preview.png',dpi=120,facecolor='white');plt.close(fig)
