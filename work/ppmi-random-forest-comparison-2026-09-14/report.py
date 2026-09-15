"""Matched nested RF/ridge comparison and paired conditional uncertainty."""
import sys,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from forest import STEP2,ARMS,evaluate
sys.path.insert(0,str(STEP2))
from reporting import weighted_metrics,bootstrap_weights,add_interval,BOOTSTRAPS,SEED,PROTOCOLS

LABELS={'original_ridge':'Original ridge','selected_ridge':'Ridge with gene selection',
        'all_genes_rf':'RF, all eligible genes','selected_rf':'RF with gene selection'}
COMPARISONS=[('selected_rf','original_ridge'),('all_genes_rf','original_ridge'),
             ('selected_ridge','original_ridge'),('selected_rf','selected_ridge'),('selected_rf','all_genes_rf')]

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
            row.update(Brier_mean=float(rr.Brier.mean()),log_loss_mean=float(rr.log_loss.mean()),train_AUROC=float(rr.train_AUROC.mean()),train_AP=float(rr.train_AP.mean()))
            primary.append(row)
        for arm,reference in COMPARISONS:
            contrasts.append(add_interval(dict(protocol=protocol,comparison=f'{arm} minus {reference}'),draws[arm]-draws[reference]))
        print(f'Bootstrap complete: {protocol}',flush=True)
    assert max(errors)<1e-12
    primary=pd.DataFrame(primary);contrasts=pd.DataFrame(contrasts)
    table('primary_summary.tsv',primary);table('paired_primary_comparisons.tsv',contrasts)
    np.savez_compressed(root/'bootstrap_metrics.npz',metric_names=np.array(['AUROC','AP']),**archive)
    differences=[]
    for arm,reference in COMPARISONS:
        a=folds[folds.arm==arm].set_index(['protocol','repeat','fold'])
        b=folds[folds.arm==reference].set_index(['protocol','repeat','fold'])
        d=(a[['AUROC','AP','Brier','log_loss']]-b[['AUROC','AP','Brier','log_loss']]).reset_index()
        d['comparison']=f'{arm} minus {reference}';differences.append(d)
    differences=pd.concat(differences);table('paired_fold_differences.tsv',differences)
    table('paired_repeat_differences.tsv',differences.groupby(['protocol','comparison','repeat'],as_index=False)[['AUROC','AP','Brier','log_loss']].mean())
    table('gene_count_selection_counts.tsv',params.groupby(['protocol','arm','k']).size().reset_index(name='outer_fold_count'))
    secondary=[]
    for (protocol,arm,repeat),part in pred.groupby(['protocol','arm','repeat']):
        yy=part.label.to_numpy();p=part.probability_PD.to_numpy()
        for mode,t in [('fixed_0_5',.5),('inner_selected',part.threshold.to_numpy())]:
            secondary.append(dict(protocol=protocol,arm=arm,repeat=repeat,predicted_prevalence=float(p.mean()),observed_prevalence=float(yy.mean()),threshold_mode=mode,**evaluate(yy,p,t)))
    secondary=pd.DataFrame(secondary);table('secondary_pooled_OOF_metrics.tsv',secondary)
    operating=secondary.groupby(['protocol','arm','threshold_mode'],as_index=False)[['sensitivity','specificity','balanced_accuracy','accuracy','Brier','AUROC','average_precision','predicted_prevalence','observed_prevalence']].mean()
    table('operating_point_summary.tsv',operating)
    plot(root,primary,repeats)
    lines=['# Step 4: random forest versus RNA-only ridge','','Same 528 participants, fixed Hallmark candidate universe and original nested folds. Gene selection and model tuning are entirely inside training.','','## Primary discrimination','','Mean within-outer-fold metrics averaged over three repeats; AP denotes average precision. Parentheses contain conditional 95% bootstrap intervals.','','| Protocol | Procedure | AUROC (95% interval) | AP (95% interval) |','| --- | --- | --- | --- |']
    for r in primary.itertuples(index=False):lines.append(f'| {r.protocol} | {LABELS[r.arm]} | {r.AUROC_mean:.4f} ({r.AUROC_ci_lower:.4f}–{r.AUROC_ci_upper:.4f}) | {r.AP_mean:.4f} ({r.AP_ci_lower:.4f}–{r.AP_ci_upper:.4f}) |')
    lines += ['','## Paired changes','','The prespecified primary contrast is selected_rf minus original_ridge. Other contrasts are descriptive controls.','','| Protocol | Comparison | AUROC difference (95% interval) | AP difference (95% interval) |','| --- | --- | --- | --- |']
    for r in contrasts.itertuples(index=False):lines.append(f'| {r.protocol} | {r.comparison} | {r.AUROC_mean:+.4f} ({r.AUROC_ci_lower:+.4f}–{r.AUROC_ci_upper:+.4f}) | {r.AP_mean:+.4f} ({r.AP_ci_lower:+.4f}–{r.AP_ci_upper:+.4f}) |')
    lines += ['','## Interpretation of the controls','','The all-gene RF comparison changes algorithm while retaining the full eligible representation. Selected ridge and selected RF share the same k options and training-only ANOVA ranking rule, but each tunes its own k; their realized selected gene sets may differ. The selected ridge grid here is 100/500/all and therefore differs from the earlier larger gene-count selection experiment.','','Brier/log loss and training AUROC/AP are saved in primary_summary.tsv. Threshold metrics at 0.5 and inner-selected thresholds are in operating_point_summary.tsv. Pooled OOF metrics use a distinct estimand from the primary within-fold metrics. No probability recalibration is applied. RF impurity importance is not a causal or unbiased gene-influence measure.','','## Verification','','All original ridge inner AUROCs, selected C, threshold, gene membership and predictions reproduced in all 30 outer folds. Training-only filtering/ranking, disjoint batch boundaries, synthetic nonlinear behavior and saved-model inference passed checks. Inner ranks and predictions are retained in inner_evidence/. No source input or prior result was changed.','',f'Paired conditional uncertainty uses {BOOTSTRAPS:,} fixed-prediction bootstrap draws per protocol. Minimum valid primary draws: {int(primary.bootstrap_valid.min()):,}; invalid draws are counted. These descriptive intervals omit refitting and the broader adaptive model-search sequence. Repeats reuse participants. Four full-data development artifacts add no new performance claim.','','![Comparison](random_forest_comparison.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    validation=dict(weighted_metric_max_error=float(max(errors)),minimum_valid_primary_draws=int(primary.bootstrap_valid.min()),checks='PASS')
    (root/'report_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    return dict(primary_results=primary.to_dict(orient='records'),paired_results=contrasts.to_dict(orient='records'),report_validation=validation)

def plot(root,primary,repeats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#247BA0','#6C80A4','#C06A27','#1B8A6B']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,protocol in enumerate(PROTOCOLS):
        s=primary[primary.protocol==protocol].set_index('arm')
        for row,metric in enumerate(['AUROC','AP']):
            ax=axes[row,j]
            for i,arm in enumerate(ARMS):
                v=s.loc[arm];point=v[f'{metric}_mean'];low=v[f'{metric}_ci_lower'];high=v[f'{metric}_ci_upper']
                ax.vlines(i,low,high,color=colors[i],lw=2);ax.scatter(i,point,color=colors[i],s=55,zorder=3)
                rr=repeats[(repeats.protocol==protocol)&(repeats.arm==arm)].sort_values('repeat')[metric]
                ax.scatter(i+np.array([-.08,0,.08]),rr,color=colors[i],s=16,alpha=.4)
                ax.annotate(f'{point:.3f}',(i,high),xytext=(0,7),textcoords='offset points',ha='center')
            ax.set_xticks(range(4),['Original\nridge','Selected\nridge','All-gene\nRF','Selected\nRF']);ax.set_xlim(-.4,3.4)
            ax.set_ylim(s[f'{metric}_ci_lower'].min()-.025,s[f'{metric}_ci_upper'].max()+.035)
            ax.set_ylabel('Mean within-fold '+('AUROC' if metric=='AUROC' else 'average precision'));ax.grid(axis='y',alpha=.15)
            ax.text(-.11,1.05,'ABCD'[row*2+j],transform=ax.transAxes,fontweight='bold',fontsize=13)
            if row==0:ax.set_title('Participant-stratified folds' if j==0 else 'Batch-grouped folds',pad=15)
    fig.suptitle('Random forest and training-only Hallmark gene selection',fontsize=15,y=.97)
    fig.text(.5,.92,'528 participants • Same candidate genes and nested folds • Three repeats',ha='center')
    fig.subplots_adjust(left=.10,right=.975,top=.83,bottom=.17,hspace=.43,wspace=.28)
    fig.text(.1,.05,'Large points: mean over repeats. Small points: repeat estimates. Bars: conditional 95% bootstrap intervals.\nSelected procedures tune gene count within training; all-gene controls retain every training-eligible candidate.',fontsize=9,linespacing=1.6)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'random_forest_comparison.{ext}',dpi=600,facecolor='white')
    fig.savefig(root/'random_forest_comparison_preview.png',dpi=120,facecolor='white');plt.close(fig)
