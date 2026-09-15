"""Validate and detach the fixed 50-Hallmark classifier experiment."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
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
from pathways import (BASE,PRIOR,FAMILIES,PENALTIES,SEED,PathwayData,choose_device,make_splits,
                      matrices,fit_model,tune,artifact,predict_artifact,evaluate,validate)

ROOT=Path(__file__).resolve().parent
SETS=BASE.parent/'ppmi-deseq2-2026-09-12/msigdb_hallmark_gobp.rds'


def save(name,obj):
    p=ROOT/name; t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); t.replace(p)


def status(state,**kw):
    save('status.json',dict(status=state,utc=datetime.now(timezone.utc).isoformat(),**kw))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def table(name,data):
    pd.DataFrame(data).to_csv(ROOT/name,sep='\t',index=False)


def prepare():
    status('PREPARING')
    old=json.loads((PRIOR/'input_sha256.json').read_text())
    for path in [BASE/'counts.npy',BASE/'gene_ids.npy',BASE/'metadata.tsv',BASE/'classifier.py']:
        assert sha(path)==old[str(path)]
    assert json.loads((PRIOR/'summary.json').read_text())['status']=='COMPLETE'
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    raw,genes=np.load(BASE/'counts.npy'),np.load(BASE/'gene_ids.npy')
    assert len(meta)==528 and meta.PATNO.is_unique and raw.shape==(528,58780)
    assert np.issubdtype(raw.dtype,np.integer) and (raw>=0).all()
    assert meta.group.value_counts().to_dict()=={'PD':358,'Control':170}
    assert np.array_equal(meta.label,meta.group.eq('PD').astype(int))
    result=subprocess.run(['/usr/local/bin/Rscript',str(ROOT/'export_hallmark.R'),str(SETS),str(ROOT/'hallmark_source.tsv')],
                          capture_output=True,text=True,check=True)
    (ROOT/'membership_export.log').write_text(result.stdout+result.stderr)
    source=pd.read_csv(ROOT/'hallmark_source.tsv',sep='\t')
    names=np.array(sorted(source.gs_name.unique()))
    assert len(names)==50 and source.db_version.unique().tolist()==['2026.1.Hs']
    memberships=source[['gs_name','ensembl_gene']].drop_duplicates().copy()
    bases=np.array([g.split('.')[0] for g in genes]); count=Counter(bases)
    lookup={base:i for i,base in enumerate(bases) if count[base]==1}
    dispositions=[]
    for row in memberships.itertuples(index=False):
        gene=row.ensembl_gene
        state='missing_identifier' if pd.isna(gene) else 'ambiguous_raw_mapping' if count[gene]>1 else 'mapped' if gene in lookup else 'unmapped'
        i=lookup[gene] if state=='mapped' else -1
        dispositions.append(dict(pathway=row.gs_name,ensembl_gene=gene,status=state,raw_index=i,Geneid=genes[i] if i>=0 else ''))
    dispositions=pd.DataFrame(dispositions)
    table('membership_mapping.tsv',dispositions)
    mapped=dispositions[dispositions.status.eq('mapped')]
    indices=np.sort(mapped.raw_index.unique())
    index_lookup={v:i for i,v in enumerate(indices)}; name_lookup={n:j for j,n in enumerate(names)}
    membership=np.zeros((len(indices),50),dtype=float)
    for r in mapped.itertuples(index=False): membership[index_lookup[r.raw_index],name_lookup[r.pathway]]=1
    assert (membership.sum(0)>=5).all()
    np.savez(ROOT/'membership.npz',indices=indices,membership=membership,names=names)
    coverage=dispositions.groupby(['pathway','status']).size().unstack(fill_value=0).reset_index()
    table('annotation_coverage.tsv',coverage)
    plan=json.loads((PRIOR/'fold_plan.json').read_text())
    assert len(plan)==30
    y,batches=meta.label.to_numpy(),meta.batch.to_numpy(str)
    for split in plan:
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        assert not np.intersect1d(tr,te).size and set(np.r_[tr,te])==set(range(528))
        assert len(np.unique(y[tr]))==len(np.unique(y[te]))==2
        if split['protocol']=='batch_grouped': assert not set(batches[tr]) & set(batches[te])
        seen=np.zeros(len(tr),dtype=int)
        for s in split['inner']:
            a,b=np.array(s['train_positions']),np.array(s['validation_positions']); seen[b]+=1
            assert not np.intersect1d(a,b).size and set(np.r_[a,b])==set(range(len(tr)))
            if split['protocol']=='batch_grouped': assert not set(batches[tr[a]]) & set(batches[tr[b]])
        assert (seen==1).all()
    save('fold_plan.json',plan)
    device=choose_device(); checks=validate(device)
    data=PathwayData(raw,genes,meta,y,indices,membership,names,device)
    tr=np.array(plan[0]['train_indices']); inner=plan[0]['inner'][0]
    data.pack(tr[np.array(inner['train_positions'])],tr[np.array(inner['validation_positions'])],'real_preflight')
    table('preflight_pathway_coverage.tsv',data.coverage)
    checked=dict(status='PASS',participants=528,PD=358,Control=170,pathways=50,mapped_gene_features=len(indices),
                 membership_dispositions={k:int(v) for k,v in dispositions.status.value_counts().items()},
                 minimum_training_members=min(r['eligible_genes'] for r in data.coverage),
                 device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU fallback',
                 sklearn=sklearn.__version__,python=sys.executable,checks=checks)
    save('preflight.json',checked)
    paths=[BASE/n for n in ['counts.npy','gene_ids.npy','metadata.tsv','classifier.py']]
    paths += [PRIOR/n for n in ['input_sha256.json','fold_plan.json','out_of_fold_predictions.tsv','outer_selected_parameters.tsv','refine.py','elastic_solver.py']]
    paths += [SETS]+[ROOT/n for n in ['README.md','export_hallmark.R','pathways.py','launch.py','predict.py','membership.npz','hallmark_source.tsv','membership_mapping.tsv','fold_plan.json']]
    save('input_sha256.json',{str(p):sha(p) for p in paths})
    return checked


def summarize(pred,metrics,prior):
    metrics=pd.DataFrame(metrics)
    table('outer_fold_metrics.tsv',metrics)
    fold=metrics[['repeat','protocol','fold','family','penalty','test_AUROC','train_AUROC']].rename(columns={'test_AUROC':'AUROC'})
    repeats=fold.groupby(['repeat','protocol','family','penalty']).agg(AUROC=('AUROC','mean'),train_AUROC=('train_AUROC','mean')).reset_index()
    repeats['train_test_gap']=repeats.train_AUROC-repeats.AUROC
    table('primary_per_repeat.tsv',repeats)
    summary=repeats.groupby(['protocol','family','penalty']).agg(AUROC_mean=('AUROC','mean'),AUROC_min=('AUROC','min'),
              AUROC_max=('AUROC','max'),train_AUROC_mean=('train_AUROC','mean'),train_test_gap_mean=('train_test_gap','mean')).reset_index()
    table('primary_summary.tsv',summary)
    secondary=[]
    for (repeat,protocol,family,penalty),group in pred.groupby(['repeat','protocol','family','penalty']):
        assert len(group)==528 and group.PATNO.is_unique
        for mode,threshold in [('fixed_0_5',.5),('inner_selected',group.threshold.to_numpy())]:
            secondary.append(dict(repeat=int(repeat),protocol=protocol,family=family,penalty=penalty,threshold_mode=mode,
                                   **evaluate(group.label.to_numpy(),group.probability_PD.to_numpy(),threshold)))
    table('secondary_pooled_OOF_metrics.tsv',secondary)
    baseline=[]
    for (repeat,protocol,fold_num,family,penalty),group in prior.groupby(['repeat','protocol','fold','family','penalty']):
        baseline.append(dict(repeat=int(repeat),protocol=protocol,fold=int(fold_num),family=family,penalty=penalty,
                             AUROC=float(roc_auc_score(group.label,group.probability_PD))))
    baseline=pd.DataFrame(baseline)
    table('preserved_gene_baseline_fold_AUROC.tsv',baseline)
    comparisons=[]
    for (repeat,protocol,fold_num,penalty),group in fold.groupby(['repeat','protocol','fold','penalty']):
        now=group.set_index('family').AUROC
        old=baseline[(baseline.repeat==repeat)&(baseline.protocol==protocol)&(baseline.fold==fold_num)&(baseline.penalty==penalty)].set_index('family').AUROC
        for name,a,b in [('pathways minus blood_clinical',now.pathways,now.blood_clinical),
                         ('pathway_combined minus blood_clinical',now.combined,now.blood_clinical),
                         ('pathways minus raw_RNA',now.pathways,old.rna),
                         ('pathway_combined minus raw_RNA_combined',now.combined,old.combined)]:
            comparisons.append(dict(repeat=int(repeat),protocol=protocol,fold=int(fold_num),penalty=penalty,comparison=name,AUROC_difference=float(a-b)))
    comparisons=pd.DataFrame(comparisons)
    table('paired_fold_comparisons.tsv',comparisons)
    paired=comparisons.groupby(['repeat','protocol','penalty','comparison']).AUROC_difference.mean().reset_index()
    table('paired_repeat_comparisons.tsv',paired)
    table('paired_summary.tsv',paired.groupby(['protocol','penalty','comparison']).AUROC_difference.agg(['mean','min','max']).reset_index())
    return summary


def train():
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str}); y=meta.label.to_numpy()
    raw,genes=np.load(BASE/'counts.npy'),np.load(BASE/'gene_ids.npy')
    mapping=np.load(ROOT/'membership.npz')
    data=PathwayData(raw,genes,meta,y,mapping['indices'],mapping['membership'],mapping['names'],choose_device())
    plan=json.loads((ROOT/'fold_plan.json').read_text())
    prior=pd.read_csv(PRIOR/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str})
    pred,metrics,params,tuning,coefficients,checks=[],[],[],[],[],[]
    for number,split in enumerate(plan,1):
        context={k:split[k] for k in ['repeat','protocol','fold']}
        status('TRAINING',completed_folds=number-1,total_folds=30,**context)
        print(f'{datetime.now(timezone.utc).isoformat()} {number}/30 {context}',flush=True)
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        inner=[(np.array(s['train_positions']),np.array(s['validation_positions'])) for s in split['inner']]
        best,rows=tune(data,tr,inner,tag=f'outer_{number}')
        tuning.extend(dict(**context,**r) for r in rows)
        pack=data.pack(tr,te,tag=f'outer_{number}_fit')
        for (family,penalty),chosen in best.items():
            x,v=matrices(pack,family); model=fit_model(x,y[tr],chosen['C'],penalty,chosen['l1_ratio'])
            p=model.predict_proba(v)[:,1]
            pred.extend(dict(**context,family=family,penalty=penalty,PATNO=meta.PATNO.iloc[i],label=int(y[i]),
                             probability_PD=float(prob),threshold=chosen['threshold']) for i,prob in zip(te,p))
            metrics.append(dict(**context,family=family,penalty=penalty,n_train=len(tr),n_test=len(te),
                                train_AUROC=float(roc_auc_score(y[tr],model.predict_proba(x)[:,1])),test_AUROC=float(roc_auc_score(y[te],p))))
            params.append(dict(**context,**chosen,features=x.shape[1],nonzero_coefficients=int(np.count_nonzero(model.coef_))))
            if family!='blood_clinical':
                offset=7 if family=='combined' else 0
                coefficients.extend(dict(**context,family=family,penalty=penalty,pathway=str(name),coefficient=float(c),nonzero=bool(c!=0)) for name,c in zip(data.names,model.coef_[0,offset:]))
            else:
                old=prior[(prior.repeat==split['repeat'])&(prior.protocol==split['protocol'])&(prior.fold==split['fold'])&
                          (prior.family=='blood_clinical')&(prior.penalty==penalty)].set_index('PATNO').loc[meta.PATNO.iloc[te]]
                np.testing.assert_allclose(p,old.probability_PD,atol=1e-8,rtol=1e-8)
                checks.append(float(np.max(np.abs(p-old.probability_PD.to_numpy()))))
            state=artifact(model,pack,family,penalty,chosen,genes)
            np.testing.assert_allclose(predict_artifact(state,raw[te[:3]],meta.iloc[te[:3]]),p[:3],atol=1e-9,rtol=1e-9)
        table('out_of_fold_predictions.tsv',pred); table('selected_parameters.tsv',params)
        table('inner_tuning_scores.tsv',tuning); table('training_pathway_coverage.tsv',data.coverage)
    pred=pd.DataFrame(pred)
    assert len(pred)==528*3*2*6 and not pred.duplicated(['repeat','protocol','family','penalty','PATNO']).any()
    status('SUMMARIZING')
    summary=summarize(pred,metrics,prior)
    table('outer_pathway_coefficients.tsv',coefficients)
    coef=pd.DataFrame(coefficients)
    stability=coef.groupby(['protocol','family','penalty','pathway']).agg(fits=('coefficient','size'),nonzero_fits=('nonzero','sum'),
               positive_fits=('coefficient',lambda x:int((x>0).sum())),negative_fits=('coefficient',lambda x:int((x<0).sum())),
               mean_coefficient=('coefficient','mean')).reset_index()
    assert (stability.fits==15).all()
    table('pathway_coefficient_stability.tsv',stability)
    save('blood_baseline_reproduction.json',dict(status='PASS',fold_comparisons=len(checks),maximum_probability_difference=max(checks)))
    status('FITTING_FINAL_DEVELOPMENT_MODELS')
    splits=make_splits(y,meta.batch.to_numpy(str),'participant_stratified',5,SEED+9901)
    save('final_tuning_folds.json',[dict(train_indices=a.tolist(),validation_indices=b.tolist()) for a,b in splits])
    best,rows=tune(data,np.arange(528),splits,tag='final')
    table('final_tuning_scores.tsv',rows)
    full=data.pack(np.arange(528),np.array([],dtype=int),tag='final_fit')
    (ROOT/'models').mkdir(exist_ok=True)
    for (family,penalty),chosen in best.items():
        x,_=matrices(full,family); model=fit_model(x,y,chosen['C'],penalty,chosen['l1_ratio'])
        state=artifact(model,full,family,penalty,chosen,genes)
        state.update(training_n=528,sklearn_version=sklearn.__version__,numpy_version=np.__version__)
        path=ROOT/'models'/f'{family}_{penalty}.joblib'; joblib.dump(state,path)
        np.testing.assert_allclose(predict_artifact(joblib.load(path),raw[:5],meta.iloc[:5]),model.predict_proba(x[:5])[:,1],atol=1e-9)
    save('final_parameters.json',list(best.values()))
    table('training_pathway_coverage.tsv',data.coverage)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    repeats=pd.read_csv(ROOT/'primary_per_repeat.tsv',sep='\t')
    for col,protocol in enumerate(['participant_stratified','batch_grouped']):
        labels=[]
        for i,(family,penalty) in enumerate((f,p) for f in FAMILIES for p in PENALTIES):
            a=repeats[(repeats.protocol==protocol)&(repeats.family==family)&(repeats.penalty==penalty)].sort_values('repeat')
            axes[col].scatter(i+np.array([-.06,0,.06]),a.AUROC)
            labels.append(f'{family}\n{penalty}')
        axes[col].axhline(.5,color='gray',ls='--')
        axes[col].set(xticks=np.arange(6),xticklabels=labels,title=protocol,ylabel='Mean within-fold AUROC per repeat',ylim=(0,1))
        axes[col].tick_params(axis='x',labelsize=7)
    fig.tight_layout(); fig.savefig(ROOT/'pathway_comparison.png',dpi=170); plt.close(fig)
    hashes=json.loads((ROOT/'input_sha256.json').read_text())
    assert all(sha(path)==value for path,value in hashes.items())
    lines=['# Hallmark pathway classifier results','',
           'All 50 predefined Hallmark pathways; 528 participants. Gene filtering/scaling, pathway score construction, model tuning and threshold selection were fitted inside training partitions. Primary AUROC is the mean of within-fold AUROCs; ranges span three repeat means and are not confidence intervals.','',
           '| Protocol | Features | Penalty | Held-out AUROC (repeat range) | Training AUROC |',
           '| --- | --- | --- | --- | ---: |']
    for r in summary.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.family} | {r.penalty} | {r.AUROC_mean:.3f} ({r.AUROC_min:.3f}–{r.AUROC_max:.3f}) | {r.train_AUROC_mean:.3f} |')
    lines += ['', 'Blood/clinical models use seven covariates, pathways use 50 expression summaries, and combined uses all 57 features. These summaries are means of training-standardized member-gene expression, not pathway activation assays or enrichment statistics. Gene membership coverage and ambiguous/unmapped identifiers are recorded.','',
              'Paired fold/repeat comparisons against the preserved raw-gene models and blood baseline are in paired_summary.tsv. Pooled OOF metrics and threshold-based results are secondary, separately labeled in secondary_pooled_OOF_metrics.tsv. Training scores are optimistic resubstitution diagnostics.','',
              'All 60 blood-baseline fold predictions reproduced prior results; all mapping, 50-set coverage, CPU/GPU, held-out perturbation, label-independence, inference, optimizer and source-integrity checks passed. Six full-data development models and prediction transforms are saved.','',
              'This is another adaptive exploratory analysis of the same cohort. Neither pathway compression nor repeated splits provide independent validation, prove RNA information beyond blood composition, or establish clinical utility. Existing significant pathways and gene shortlists were not used to select features.','',
              '![Pathway comparison](pathway_comparison.png)','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    save('summary.json',dict(status='COMPLETE',participants=528,pathways=50,preflight=json.loads((ROOT/'preflight.json').read_text()),
                             primary_results=summary.to_dict(orient='records'),source_hashes_unchanged=True,
                             blood_baseline_reproduction='PASS',artifact_roundtrips='PASS',external_validation=False))
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
                                         stderr=subprocess.STDOUT,env=env,cwd=ROOT.parent.parent,start_new_session=True)
                save('job.json',dict(pid=job.pid,device=checked['device_name'],log=str(ROOT/'run.log')))
                print(json.dumps(dict(launched_pid=job.pid,preflight=checked),indent=2))
    except Exception:
        status('FAILED',error=traceback.format_exc()); raise
