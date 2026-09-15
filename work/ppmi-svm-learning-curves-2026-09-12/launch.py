"""Launch controlled nonlinear comparison and nested learning curves."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from svm_analysis import (BASE,FAMILIES,ALGORITHMS,FRACTIONS,SEED,FoldData,choose_device,make_splits,
                          matrices,artifact,fit_model,score,tune,subset_plan,predict_artifact,validate_svm)

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'ppmi-classifier-refinement-retry-2026-09-12'


def save(name,obj):
    p=ROOT/name; temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); temp.replace(p)


def status(state,**kwargs):
    save('status.json',dict(status=state,utc=datetime.now(timezone.utc).isoformat(),**kwargs))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def prepare():
    status('PREPARING')
    old_hashes=json.loads((PRIOR/'input_sha256.json').read_text())
    for path in [BASE/'counts.npy',BASE/'gene_ids.npy',BASE/'metadata.tsv',BASE/'classifier.py']:
        assert sha(path)==old_hashes[str(path)]
    assert json.loads((PRIOR/'summary.json').read_text())['status']=='COMPLETE'
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    genes,raw=np.load(BASE/'gene_ids.npy'),np.load(BASE/'counts.npy',mmap_mode='r')
    assert len(meta)==528 and meta.PATNO.is_unique and raw.shape==(528,58780)
    assert len(np.unique(genes))==58780 and np.issubdtype(raw.dtype,np.integer) and (raw>=0).all()
    assert meta.group.value_counts().to_dict()=={'PD':358,'Control':170}
    assert np.array_equal(meta.label.to_numpy(),meta.group.eq('PD').astype(int).to_numpy())
    y,batches=meta.label.to_numpy(),meta.batch.to_numpy(str)
    original=json.loads((PRIOR/'fold_plan.json').read_text())
    assert len(original)==30
    plan=[]
    for split in original:
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        assert set(np.r_[tr,te])==set(range(528)) and not np.intersect1d(tr,te).size
        if split['protocol']=='batch_grouped':
            assert not set(batches[tr]) & set(batches[te])
        subsets=subset_plan(tr,y,batches,split['protocol'],split['seed']+100*split['fold'],split['inner'])
        assert subsets[-1]['inner']==split['inner']
        plan.append(dict(**split,learning_subsets=subsets))
    save('fold_and_learning_plan.json',plan)
    device=choose_device(); checks=validate_svm(device)
    preflight=dict(status='PASS',participants=528,PD=358,Control=170,raw_genes=58780,repeats=3,
                   outer_folds=30,learning_tasks=120,families=FAMILIES,algorithms=ALGORITHMS,
                   device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU fallback',
                   sklearn=sklearn.__version__,torch=torch.__version__,python=sys.executable,checks=checks)
    save('preflight.json',preflight)
    paths=[BASE/n for n in ['counts.npy','gene_ids.npy','metadata.tsv','classifier.py']]
    paths += [PRIOR/n for n in ['input_sha256.json','fold_plan.json','out_of_fold_predictions.tsv','outer_selected_parameters.tsv','summary.json']]
    paths += [ROOT/n for n in ['README.md','svm_analysis.py','launch.py','predict.py','fold_and_learning_plan.json']]
    save('input_sha256.json',{str(p):sha(p) for p in paths})
    return preflight


def write_table(name,rows):
    pd.DataFrame(rows).to_csv(ROOT/name,sep='\t',index=False)


def summarize_full(curves,prior):
    full=curves[curves.fraction.eq(1.)][['repeat','protocol','fold','family','algorithm','test_AUROC']].copy()
    full=full.rename(columns={'test_AUROC':'AUROC'})
    elastic=[]
    for (repeat,protocol,fold,family),group in prior[prior.penalty.eq('elasticnet')].groupby(['repeat','protocol','fold','family']):
        elastic.append(dict(repeat=int(repeat),protocol=protocol,fold=int(fold),family=family,algorithm='elasticnet',
                            AUROC=float(roc_auc_score(group.label,group.probability_PD))))
    full=pd.concat([full,pd.DataFrame(elastic)],ignore_index=True)
    assert len(full)==270 and not full.duplicated(['repeat','protocol','fold','family','algorithm']).any()
    write_table('full_size_fold_AUROC.tsv',full)
    repeats=full.groupby(['repeat','protocol','family','algorithm']).AUROC.mean().reset_index(name='mean_fold_AUROC')
    write_table('full_size_repeat_AUROC.tsv',repeats)
    summary=repeats.groupby(['protocol','family','algorithm']).mean_fold_AUROC.agg(['mean','min','max']).reset_index()
    write_table('full_size_comparison.tsv',summary)
    paired=[]
    for (repeat,protocol,fold),part in full.groupby(['repeat','protocol','fold']):
        scores=part.set_index(['family','algorithm']).AUROC
        for family in FAMILIES:
            for comparator in ['ridge','elasticnet']:
                paired.append(dict(repeat=int(repeat),protocol=protocol,fold=int(fold),
                       comparison=f'{family}: rbf_svm minus {comparator}',
                       AUROC_difference=float(scores.loc[(family,'rbf_svm')]-scores.loc[(family,comparator)])))
        for algorithm in ['ridge','elasticnet','rbf_svm']:
            paired.append(dict(repeat=int(repeat),protocol=protocol,fold=int(fold),
                       comparison=f'{algorithm}: combined minus blood_clinical',
                       AUROC_difference=float(scores.loc[('combined',algorithm)]-scores.loc[('blood_clinical',algorithm)])))
    pairs=pd.DataFrame(paired)
    write_table('paired_fold_comparisons.tsv',pairs)
    pair_repeats=pairs.groupby(['repeat','protocol','comparison']).AUROC_difference.mean().reset_index()
    write_table('paired_repeat_comparisons.tsv',pair_repeats)
    return summary


def summarize_learning(curves):
    keys=['repeat','protocol','family','algorithm','fraction']
    repeats=curves.groupby(keys).agg(n_train=('n_train','mean'),train_AUROC=('train_AUROC','mean'),test_AUROC=('test_AUROC','mean')).reset_index()
    repeats['train_test_gap']=repeats.train_AUROC-repeats.test_AUROC
    write_table('learning_curves_by_repeat.tsv',repeats)
    summary=repeats.groupby(keys[1:]).agg(n_train_mean=('n_train','mean'),train_AUROC_mean=('train_AUROC','mean'),
             test_AUROC_mean=('test_AUROC','mean'),test_AUROC_min=('test_AUROC','min'),test_AUROC_max=('test_AUROC','max'),
             train_test_gap_mean=('train_test_gap','mean')).reset_index()
    write_table('learning_curve_summary.tsv',summary)
    changes=[]
    for (repeat,protocol,family,algorithm),part in repeats.groupby(keys[:4]):
        a=part.set_index('fraction')
        changes.append(dict(repeat=int(repeat),protocol=protocol,family=family,algorithm=algorithm,
                            test_AUROC_change_25_to_100=float(a.loc[1.,'test_AUROC']-a.loc[.25,'test_AUROC']),
                            test_AUROC_change_75_to_100=float(a.loc[1.,'test_AUROC']-a.loc[.75,'test_AUROC'])))
    write_table('observed_learning_changes.tsv',changes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,12),sharey=True)
    colors={'ridge':'#2478a4','rbf_svm':'#bd5034'}
    for row,family in enumerate(FAMILIES):
        for col,protocol in enumerate(['participant_stratified','batch_grouped']):
            ax=axes[row,col]
            for algorithm in ALGORITHMS:
                a=summary[(summary.family==family)&(summary.protocol==protocol)&(summary.algorithm==algorithm)].sort_values('fraction')
                ax.plot(a.n_train_mean,a.test_AUROC_mean,'o-',color=colors[algorithm],label=algorithm+' held-out')
                ax.fill_between(a.n_train_mean.to_numpy(),a.test_AUROC_min.to_numpy(),a.test_AUROC_max.to_numpy(),color=colors[algorithm],alpha=.12)
                ax.plot(a.n_train_mean,a.train_AUROC_mean,'--',color=colors[algorithm],label=algorithm+' training')
            ax.axhline(.5,color='gray',ls=':')
            ax.set(title=f'{protocol}: {family}',xlabel='Mean actual outer-training participants',ylabel='Mean within-fold AUROC',ylim=(0,1.02))
            ax.legend(fontsize=8)
    fig.suptitle('Nested learning curves — shading is the range of three repeat means, not a confidence interval',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(ROOT/'learning_curves.png',dpi=170); plt.close(fig)
    return summary


def train():
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    raw,genes=np.load(BASE/'counts.npy'),np.load(BASE/'gene_ids.npy')
    y,batches=meta.label.to_numpy(),meta.batch.to_numpy(str)
    data=FoldData(raw,genes,meta,y,choose_device())
    plan=json.loads((ROOT/'fold_and_learning_plan.json').read_text())
    prior=pd.read_csv(PRIOR/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str})
    prior=prior[prior.family.isin(FAMILIES)]
    prior_params=pd.read_csv(PRIOR/'outer_selected_parameters.tsv',sep='\t')
    curves,predictions,tuning,parameters,features,checks=[],[],[],[],[],[]
    # Finish full-size comparison first, then smaller training subsets.
    tasks=[(split,subset) for fraction in [1.,.25,.5,.75] for split in plan
           for subset in split['learning_subsets'] if subset['fraction']==fraction]
    for number,(split,subset) in enumerate(tasks,1):
        context={key:split[key] for key in ['repeat','protocol','fold']}
        fraction=subset['fraction']
        status('TRAINING',completed_tasks=number-1,total_tasks=120,fraction=fraction,**context)
        print(f'{datetime.now(timezone.utc).isoformat()} {number}/120 fraction={fraction} {context}',flush=True)
        tr,te=np.array(subset['train_indices']),np.array(split['test_indices'])
        assert not np.intersect1d(tr,te).size
        inner=[(np.array(s['train_positions']),np.array(s['validation_positions'])) for s in subset['inner']]
        best,rows=tune(data,tr,inner)
        tuning.extend(dict(**context,fraction=fraction,**row) for row in rows)
        pack=data.pack(tr,te)
        for (family,algorithm),chosen in best.items():
            x,v=matrices(pack,family,chosen['k'])
            model=fit_model(x,y[tr],algorithm,chosen['C'],chosen['gamma_multiplier'])
            train_score,test_score=score(model,x,algorithm),score(model,v,algorithm)
            curves.append(dict(**context,fraction=fraction,family=family,algorithm=algorithm,n_train=len(tr),n_test=len(te),
                               train_groups=subset['n_train_groups'],train_PD=int(y[tr].sum()),train_Control=int((y[tr]==0).sum()),
                               train_AUROC=float(roc_auc_score(y[tr],train_score)),test_AUROC=float(roc_auc_score(y[te],test_score))))
            parameters.append(dict(**context,fraction=fraction,**chosen,actual_features=x.shape[1],
                                   actual_gamma=chosen['gamma_multiplier']/x.shape[1] if algorithm=='rbf_svm' else 0.,
                                   support_vectors=int(model.n_support_.sum()) if algorithm=='rbf_svm' else 0))
            if fraction==1.:
                predictions.extend(dict(**context,family=family,algorithm=algorithm,sample_index=int(i),PATNO=meta.PATNO.iloc[i],
                                        label=int(y[i]),score=float(s),score_type='probability_PD' if algorithm=='ridge' else 'uncalibrated_decision_margin')
                                   for i,s in zip(te,test_score))
                if algorithm=='ridge':
                    old=prior[(prior.repeat==split['repeat'])&(prior.protocol==split['protocol'])&(prior.fold==split['fold'])&
                              (prior.family==family)&(prior.penalty=='ridge')].set_index('PATNO').loc[meta.PATNO.iloc[te]]
                    np.testing.assert_allclose(test_score,old.probability_PD,atol=1e-8,rtol=1e-8)
                    old_param=prior_params[(prior_params.repeat==split['repeat'])&(prior_params.protocol==split['protocol'])&
                              (prior_params.fold==split['fold'])&(prior_params.family==family)&(prior_params.penalty=='ridge')].iloc[0]
                    assert chosen['k']==old_param.k and chosen['C']==old_param.C
                    checks.append(dict(**context,family=family,maximum_probability_difference=float(np.max(np.abs(test_score-old.probability_PD.to_numpy())))))
                state=artifact(model,pack,family,chosen['k'],genes); state['algorithm']=algorithm
                np.testing.assert_allclose(predict_artifact(state,raw[te[:3]],meta.iloc[te[:3]]),test_score[:3],atol=1e-9,rtol=1e-9)
                if family in ['rna','combined']:
                    features.extend(dict(**context,family=family,algorithm=algorithm,Geneid=gene) for gene in state['selected_gene_ids'])
        write_table('learning_curve_fold_metrics.tsv',curves)
        write_table('selected_parameters.tsv',parameters)
        write_table('inner_tuning_scores.tsv',tuning)
        if fraction==1.:
            write_table('full_size_out_of_fold_scores.tsv',predictions)
        if number==30:
            summarize_full(pd.DataFrame(curves),prior)
            save('benchmark_reproduction.json',dict(status='PASS',comparisons=len(checks),maximum_probability_difference=max(r['maximum_probability_difference'] for r in checks)))
    curves=pd.DataFrame(curves)
    assert len(curves)==720 and not curves.duplicated(['repeat','protocol','fold','fraction','family','algorithm']).any()
    pred=pd.DataFrame(predictions)
    assert len(pred)==528*3*2*3*2 and not pred.duplicated(['repeat','protocol','family','algorithm','PATNO']).any()
    write_table('full_size_selected_genes.tsv',features)
    write_table('ridge_reproduction_by_fold.tsv',checks)
    status('SUMMARIZING')
    full_summary=summarize_full(curves,prior)
    learning_summary=summarize_learning(curves)
    status('FITTING_FINAL_SVM_ARTIFACTS')
    splits=make_splits(y,batches,'participant_stratified',5,SEED+9901)
    save('final_tuning_folds.json',[dict(train_indices=a.tolist(),validation_indices=b.tolist()) for a,b in splits])
    best,rows=tune(data,np.arange(528),splits,algorithms=['rbf_svm'])
    write_table('final_svm_tuning.tsv',rows)
    full=data.pack(np.arange(528),np.array([],dtype=int))
    (ROOT/'models').mkdir(exist_ok=True)
    for (family,algorithm),chosen in best.items():
        x,_=matrices(full,family,chosen['k'])
        model=fit_model(x,y,algorithm,chosen['C'],chosen['gamma_multiplier'])
        state=artifact(model,full,family,chosen['k'],genes)
        state.update(algorithm=algorithm,score_type='uncalibrated_decision_margin',threshold=0.,
                     threshold_description='Native SVM boundary; not tuned or clinically validated',
                     hyperparameters=chosen,training_n=528,sklearn_version=sklearn.__version__)
        path=ROOT/'models'/f'{family}_rbf_svm.joblib'
        joblib.dump(state,path)
        np.testing.assert_allclose(predict_artifact(joblib.load(path),raw[:5],meta.iloc[:5]),score(model,x[:5],algorithm),atol=1e-9)
    save('final_parameters.json',list(best.values()))
    hashes=json.loads((ROOT/'input_sha256.json').read_text())
    assert all(sha(path)==expected for path,expected in hashes.items()),'Source changed during analysis'
    lines=['# Controlled RBF-SVM comparison and learning curves','',
           '528 participants, three repeats, five outer folds per protocol. Primary AUROC is the mean of within-fold AUROCs, with min–max of three repeat means. This avoids pooling incomparable SVM margin scales. The previous pooled ~0.61 AUROC is not numerically identical to this estimand.','',
           '| Protocol | Feature set | Algorithm | Mean within-fold AUROC (repeat range) |',
           '| --- | --- | --- | --- |']
    for r in full_summary.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.family} | {r.algorithm} | {r.mean:.3f} ({r.min:.3f}–{r.max:.3f}) |')
    lines += ['', '## Learning curves','',
              'All feature selection, scaling and tuning were repeated inside each training subset. The outer test participants stayed fixed. Batch subsets preserve whole batches, so actual training sizes vary. Shading is the range of repeat means, not a confidence interval.','',
              '| Protocol | Features | Algorithm | Held-out AUROC at 25% | At 50% | At 75% | At 100% | Full-size training AUROC |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for (protocol,family,algorithm),part in learning_summary.groupby(['protocol','family','algorithm']):
        a=part.set_index('fraction')
        lines.append(f'| {protocol} | {family} | {algorithm} | {a.loc[.25,"test_AUROC_mean"]:.3f} | {a.loc[.5,"test_AUROC_mean"]:.3f} | {a.loc[.75,"test_AUROC_mean"]:.3f} | {a.loc[1.,"test_AUROC_mean"]:.3f} | {a.loc[1.,"train_AUROC_mean"]:.3f} |')
    lines += ['', 'A training–test gap describes overfitting risk; an upward held-out trend can motivate more data but does not guarantee improvement or justify extrapolating a target AUROC. The apparent shape also reflects changing selected features and tuned parameters.','',
              'All full-size ridge probabilities and selected parameters reproduced the prior benchmark. SVM convergence, synthetic nonlinear behavior, held-out perturbation, raw-count inference/serialization, fold/subset checks and source hashes passed. Three final SVM development artifacts are saved; their margins are not calibrated probabilities.','',
              'Repeated use of the same cohort and adaptive development decisions limit interpretation. This is not external validation. No earlier two-gene/DE shortlist was reused, no SVM threshold was optimized, and no clinical performance claim follows from these comparisons. Paired fold/repeat differences and complete tuning evidence are saved alongside the report.','',
              '![Learning curves](learning_curves.png)','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    save('summary.json',dict(status='COMPLETE',participants=528,repeats=3,learning_tasks=120,
                             full_size_comparison=full_summary.to_dict(orient='records'),
                             ridge_reproduction='PASS',artifact_inference='PASS',source_hashes_unchanged=True,external_validation=False))
    status('COMPLETE',report='RESULTS.md')


if __name__=='__main__':
    if '--run' not in sys.argv and (ROOT/'status.json').exists():
        raise SystemExit('Existing run; refusing duplicate launch.')
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            if '--run' in sys.argv:
                train()
            else:
                checked=prepare()
                env=os.environ.copy()
                env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
                (ROOT/'matplotlib_cache').mkdir(exist_ok=True)
                status('LAUNCHING')
                with (ROOT/'run.log').open('ab',buffering=0) as log:
                    job=subprocess.Popen([sys.executable,'-u',str(ROOT/'launch.py'),'--run'],stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent.parent,env=env,start_new_session=True)
                save('job.json',dict(pid=job.pid,device=checked['device_name'],log=str(ROOT/'run.log')))
                print(json.dumps(dict(launched_pid=job.pid,preflight=checked),indent=2))
    except Exception:
        status('FAILED',error=traceback.format_exc())
        raise
