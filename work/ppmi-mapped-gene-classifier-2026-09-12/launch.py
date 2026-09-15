"""Launch RNA-only classifiers on the fixed 4,376-gene Hallmark union."""
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
from mapped_genes import (BASE,PRIOR,HALLMARK,PENALTIES,SEED,MappedData,choose_device,make_splits,
                          fit_model,tune,artifact,predict_artifact,evaluate,validate)

ROOT=Path(__file__).resolve().parent


def save(name,value):
    p=ROOT/name; tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(p)


def status(state,**kw):
    save('status.json',dict(status=state,utc=datetime.now(timezone.utc).isoformat(),**kw))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def table(name,rows):
    pd.DataFrame(rows).to_csv(ROOT/name,sep='\t',index=False)


def prepare():
    status('PREPARING')
    old=json.loads((HALLMARK/'input_sha256.json').read_text())
    for path in [BASE/'counts.npy',BASE/'gene_ids.npy',BASE/'metadata.tsv',HALLMARK/'membership.npz']:
        assert sha(path)==old[str(path)]
    assert json.loads((HALLMARK/'summary.json').read_text())['status']=='COMPLETE'
    raw,genes=np.load(BASE/'counts.npy'),np.load(BASE/'gene_ids.npy')
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    assert len(meta)==528 and meta.PATNO.is_unique and raw.shape==(528,58780)
    assert np.issubdtype(raw.dtype,np.integer) and (raw>=0).all()
    assert meta.group.value_counts().to_dict()=={'PD':358,'Control':170}
    assert np.array_equal(meta.label,meta.group.eq('PD').astype(int))
    mapping=np.load(HALLMARK/'membership.npz'); indices=mapping['indices']
    assert len(indices)==len(np.unique(indices))==4376
    assert mapping['membership'].shape==(4376,50) and (mapping['membership'].sum(1)>0).all()
    table('candidate_genes.tsv',dict(Geneid=genes[indices],raw_index=indices,Hallmark_memberships=mapping['membership'].sum(1).astype(int)))
    plan=json.loads((PRIOR/'fold_plan.json').read_text()); assert len(plan)==30
    y,batches=meta.label.to_numpy(),meta.batch.to_numpy(str)
    for split in plan:
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        assert not np.intersect1d(tr,te).size and set(np.r_[tr,te])==set(range(528))
        if split['protocol']=='batch_grouped': assert not set(batches[tr]) & set(batches[te])
        seen=np.zeros(len(tr),dtype=int)
        for inner in split['inner']:
            a,b=np.array(inner['train_positions']),np.array(inner['validation_positions']); seen[b]+=1
            assert not np.intersect1d(a,b).size and set(np.r_[a,b])==set(range(len(tr)))
            assert len(np.unique(y[tr[a]]))==len(np.unique(y[tr[b]]))==2
            if split['protocol']=='batch_grouped': assert not set(batches[tr[a]]) & set(batches[tr[b]])
        assert (seen==1).all()
    save('fold_plan.json',plan)
    device=choose_device(); checks=validate(device)
    data=MappedData(raw,genes,y,indices,device)
    tr=np.array(plan[0]['train_indices']); inner=plan[0]['inner'][0]
    pack=data.pack(tr[np.array(inner['train_positions'])],tr[np.array(inner['validation_positions'])])
    checked=dict(status='PASS',participants=528,PD=358,Control=170,candidate_genes=4376,
                 first_inner_eligible_genes=pack['train'].shape[1],outer_folds=30,penalties=PENALTIES,
                 device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU fallback',
                 sklearn=sklearn.__version__,python=sys.executable,checks=checks)
    save('preflight.json',checked)
    paths=[BASE/n for n in ['counts.npy','gene_ids.npy','metadata.tsv','classifier.py']]
    paths += [PRIOR/n for n in ['fold_plan.json','out_of_fold_predictions.tsv','refine.py','elastic_solver.py']]
    paths += [HALLMARK/n for n in ['membership.npz','membership_mapping.tsv','out_of_fold_predictions.tsv','input_sha256.json']]
    paths += [ROOT/n for n in ['README.md','mapped_genes.py','launch.py','predict.py','candidate_genes.tsv','fold_plan.json']]
    save('input_sha256.json',{str(p):sha(p) for p in paths})
    return checked


def summarize(pred,fold_metrics):
    fold=pd.DataFrame(fold_metrics)
    table('outer_fold_metrics.tsv',fold)
    repeats=fold.groupby(['repeat','protocol','penalty']).agg(AUROC=('test_AUROC','mean'),train_AUROC=('train_AUROC','mean'),
                        eligible_genes=('eligible_genes','mean'),nonzero_genes=('nonzero_genes','mean')).reset_index()
    repeats['train_test_gap']=repeats.train_AUROC-repeats.AUROC
    table('primary_per_repeat.tsv',repeats)
    summary=repeats.groupby(['protocol','penalty']).agg(AUROC_mean=('AUROC','mean'),AUROC_min=('AUROC','min'),
              AUROC_max=('AUROC','max'),train_AUROC_mean=('train_AUROC','mean'),train_test_gap_mean=('train_test_gap','mean'),
              mean_eligible_genes=('eligible_genes','mean'),mean_nonzero_genes=('nonzero_genes','mean')).reset_index()
    table('primary_summary.tsv',summary)
    secondary=[]
    for (repeat,protocol,penalty),group in pred.groupby(['repeat','protocol','penalty']):
        assert len(group)==528 and group.PATNO.is_unique
        for mode,threshold in [('fixed_0_5',.5),('inner_selected',group.threshold.to_numpy())]:
            secondary.append(dict(repeat=int(repeat),protocol=protocol,penalty=penalty,threshold_mode=mode,
                 **evaluate(group.label.to_numpy(),group.probability_PD.to_numpy(),threshold)))
    table('secondary_pooled_OOF_metrics.tsv',secondary)
    baselines=[]
    for folder,families in [(PRIOR,{'rna':'prior_raw_RNA','blood_clinical':'blood_clinical'}),(HALLMARK,{'pathways':'pathway_scores'})]:
        old=pd.read_csv(folder/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str})
        for (repeat,protocol,fold_num,family,penalty),group in old[old.family.isin(families)].groupby(['repeat','protocol','fold','family','penalty']):
            baselines.append(dict(repeat=int(repeat),protocol=protocol,fold=int(fold_num),penalty=penalty,benchmark=families[family],
                                  AUROC=float(roc_auc_score(group.label,group.probability_PD))))
    old=pd.DataFrame(baselines); table('benchmark_fold_AUROC.tsv',old)
    paired=old.merge(fold[['repeat','protocol','fold','penalty','test_AUROC']],on=['repeat','protocol','fold','penalty'],validate='many_to_one')
    paired['AUROC_difference']=paired.test_AUROC-paired.AUROC
    table('paired_fold_comparisons.tsv',paired)
    repeat_paired=paired.groupby(['repeat','protocol','penalty','benchmark']).AUROC_difference.mean().reset_index()
    table('paired_repeat_comparisons.tsv',repeat_paired)
    table('paired_summary.tsv',repeat_paired.groupby(['protocol','penalty','benchmark']).AUROC_difference.agg(['mean','min','max']).reset_index())
    return summary


def train():
    raw,genes=np.load(BASE/'counts.npy'),np.load(BASE/'gene_ids.npy')
    indices=np.load(HALLMARK/'membership.npz')['indices']
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str}); y=meta.label.to_numpy()
    data=MappedData(raw,genes,y,indices,choose_device())
    plan=json.loads((ROOT/'fold_plan.json').read_text())
    predictions,metrics,params,tuning,coefficients=[],[],[],[],[]
    for number,split in enumerate(plan,1):
        context={k:split[k] for k in ['repeat','protocol','fold']}
        status('TRAINING',completed_folds=number-1,total_folds=30,**context)
        print(f'{datetime.now(timezone.utc).isoformat()} {number}/30 {context}',flush=True)
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        inner=[(np.array(s['train_positions']),np.array(s['validation_positions'])) for s in split['inner']]
        best,rows=tune(data,tr,inner); tuning.extend(dict(**context,**r) for r in rows)
        pack=data.pack(tr,te)
        assert set(pack['state']['indices'])<=set(indices)
        for penalty,chosen in best.items():
            model=fit_model(pack['train'],y[tr],chosen['C'],penalty,chosen['l1_ratio'])
            p=model.predict_proba(pack['test'])[:,1]
            predictions.extend(dict(**context,penalty=penalty,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(prob),
                                    threshold=chosen['threshold']) for i,prob in zip(te,p))
            metrics.append(dict(**context,penalty=penalty,n_train=len(tr),n_test=len(te),eligible_genes=pack['train'].shape[1],
                                nonzero_genes=int(np.count_nonzero(model.coef_)),
                                train_AUROC=float(roc_auc_score(y[tr],model.predict_proba(pack['train'])[:,1])),test_AUROC=float(roc_auc_score(y[te],p))))
            params.append(dict(**context,**chosen,eligible_genes=pack['train'].shape[1],
                               optimizer_iterations=int(getattr(model,'total_optimizer_iterations_',model.n_iter_[0]))))
            state=artifact(model,pack,chosen,genes)
            np.testing.assert_allclose(predict_artifact(state,raw[te[:3]]),p[:3],atol=1e-9,rtol=1e-9)
            coefficients.extend(dict(**context,penalty=penalty,Geneid=g,coefficient=float(c),nonzero=bool(c!=0)) for g,c in zip(state['selected_gene_ids'],model.coef_[0]))
        table('out_of_fold_predictions.tsv',predictions); table('selected_parameters.tsv',params); table('inner_tuning_scores.tsv',tuning)
    pred=pd.DataFrame(predictions)
    assert len(pred)==528*3*2*2 and not pred.duplicated(['repeat','protocol','penalty','PATNO']).any()
    status('SUMMARIZING')
    summary=summarize(pred,metrics)
    coef=pd.DataFrame(coefficients); table('outer_gene_coefficients.tsv',coef)
    stable=coef.groupby(['protocol','penalty','Geneid']).agg(eligible_fits=('coefficient','size'),nonzero_fits=('nonzero','sum'),
                  positive_fits=('coefficient',lambda x:int((x>0).sum())),negative_fits=('coefficient',lambda x:int((x<0).sum())),
                  mean_coefficient_when_eligible=('coefficient','mean')).reset_index()
    stable['total_fits']=15; assert (stable.eligible_fits<=15).all()
    table('gene_retention.tsv',stable)
    status('FITTING_FINAL_DEVELOPMENT_MODELS')
    splits=make_splits(y,meta.batch.to_numpy(str),'participant_stratified',5,SEED+9901)
    save('final_tuning_folds.json',[dict(train_indices=a.tolist(),validation_indices=b.tolist()) for a,b in splits])
    best,rows=tune(data,np.arange(528),splits); table('final_tuning_scores.tsv',rows)
    full=data.pack(np.arange(528),np.array([],dtype=int)); (ROOT/'models').mkdir(exist_ok=True)
    for penalty,chosen in best.items():
        model=fit_model(full['train'],y,chosen['C'],penalty,chosen['l1_ratio'])
        state=artifact(model,full,chosen,genes)
        state.update(training_n=528,candidate_gene_ids=genes[indices],sklearn_version=sklearn.__version__,numpy_version=np.__version__)
        path=ROOT/'models'/f'mapped_RNA_{penalty}.joblib'; joblib.dump(state,path)
        np.testing.assert_allclose(predict_artifact(joblib.load(path),raw[:5]),model.predict_proba(full['train'][:5])[:,1],atol=1e-9)
    save('final_parameters.json',list(best.values()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    repeats=pd.read_csv(ROOT/'primary_per_repeat.tsv',sep='\t')
    fig,ax=plt.subplots(figsize=(8,5))
    labels=[]
    for i,(protocol,penalty) in enumerate((p,r) for p in ['participant_stratified','batch_grouped'] for r in PENALTIES):
        a=repeats[(repeats.protocol==protocol)&(repeats.penalty==penalty)].sort_values('repeat')
        ax.scatter(i+np.array([-.06,0,.06]),a.AUROC); labels.append(f'{protocol}\n{penalty}')
    ax.axhline(.5,color='gray',ls='--'); ax.set(xticks=np.arange(4),xticklabels=labels,ylim=(0,1),ylabel='Mean within-fold AUROC per repeat')
    ax.tick_params(axis='x',labelsize=8); fig.tight_layout(); fig.savefig(ROOT/'mapped_gene_comparison.png',dpi=170); plt.close(fig)
    hashes=json.loads((ROOT/'input_sha256.json').read_text())
    assert all(sha(p)==v for p,v in hashes.items())
    lines=['# RNA-only classifier on the 4,376 Hallmark-mapped genes','',
           '528 participants; all eligible individual mapped genes, no pathway averaging or supervised top-k selection. The fixed candidate universe has 4,376 genes; training-fold detection/variance filtering determines the actual feature count. No clinical or blood covariate enters these models.','',
           '| Protocol | Penalty | Held-out AUROC (repeat range) | Training AUROC | Mean eligible genes | Mean nonzero genes |',
           '| --- | --- | --- | ---: | ---: | ---: |']
    for r in summary.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.penalty} | {r.AUROC_mean:.3f} ({r.AUROC_min:.3f}–{r.AUROC_max:.3f}) | {r.train_AUROC_mean:.3f} | {r.mean_eligible_genes:.1f} | {r.mean_nonzero_genes:.1f} |')
    lines += ['', 'Primary AUROC is the mean of within-fold AUROCs; ranges span three repeat means and are not confidence intervals. Original raw-RNA, blood/clinical and pathway comparisons use identical test folds; paired differences are in paired_summary.tsv. Secondary pooled metrics and threshold results are separately labeled.','',
              'The Hallmark union is an external biological restriction, not a set chosen by PD association strength. However, the decision to try it followed earlier results. The prior raw-RNA approach also used supervised top-k selection, so this experiment compares complete pipelines rather than isolating the effect of membership restriction alone.','',
              'All mapping, cohort, count, fold, CPU/GPU, label-independence, holdout-perturbation, inference, convergence and source-hash checks passed. Two final development artifacts are saved. This remains exploratory internal validation; no external replication or clinical utility is established.','',
              '![Mapped-gene comparison](mapped_gene_comparison.png)','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    save('summary.json',dict(status='COMPLETE',participants=528,candidate_genes=4376,final_eligible_genes=full['train'].shape[1],
                             primary_results=summary.to_dict(orient='records'),validation='PASS',source_hashes_unchanged=True,external_validation=False))
    status('COMPLETE',report='RESULTS.md')


if __name__=='__main__':
    if '--run' not in sys.argv and (ROOT/'status.json').exists(): raise SystemExit('Existing run; refusing duplicate launch')
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            if '--run' in sys.argv: train()
            else:
                checked=prepare(); env=os.environ.copy()
                env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
                (ROOT/'matplotlib_cache').mkdir(exist_ok=True); status('LAUNCHING')
                with (ROOT/'run.log').open('ab',buffering=0) as log:
                    job=subprocess.Popen([sys.executable,'-u',str(ROOT/'launch.py'),'--run'],stdin=subprocess.DEVNULL,stdout=log,
                                         stderr=subprocess.STDOUT,cwd=ROOT.parent.parent,env=env,start_new_session=True)
                save('job.json',dict(pid=job.pid,device=checked['device_name'],log=str(ROOT/'run.log')))
                print(json.dumps(dict(launched_pid=job.pid,preflight=checked),indent=2))
    except Exception:
        status('FAILED',error=traceback.format_exc()); raise
