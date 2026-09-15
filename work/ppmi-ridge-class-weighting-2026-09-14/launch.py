"""Execute the prespecified RNA ridge weighting comparison."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
os.environ['MPLCONFIGDIR']=str(HERE/'matplotlib_cache')
import sys,json,time,hashlib,traceback
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch,joblib,sklearn
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from weighting import *

def save(name,value):
    p=HERE/name;temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)

def status(state,**kw):save('status.json',dict(status=state,utc=datetime.now(timezone.utc).isoformat(),**kw))
def table(name,rows):pd.DataFrame(rows).to_csv(HERE/name,sep='\t',index=False)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def prepare(device):
    previous=json.loads((STEP2/'input_sha256.json').read_text())
    assert all(sha(p)==v for p,v in previous.items())
    inputs=list(previous)+[str(STEP2/n) for n in ['input_sha256.json','reporting.py']]
    inputs += [str(HERE/n) for n in ['README.md','weighting.py','launch.py','report.py']]
    save('input_sha256.json',{p:sha(p) for p in inputs})
    raw=np.load(BASE/'counts.npy');genes=np.load(BASE/'gene_ids.npy')
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str});y=meta.label.to_numpy()
    assert raw.shape==(528,58780) and len(np.unique(genes))==58780
    assert np.issubdtype(raw.dtype,np.integer) and (raw>=0).all()
    assert len(meta)==528 and meta.PATNO.is_unique and np.bincount(y).tolist()==[170,358]
    assert np.array_equal(y,meta.group.eq('PD').astype(int))
    indices=np.load(HALLMARK/'membership.npz')['indices']
    candidates=pd.read_csv(MAPPED/'candidate_genes.tsv',sep='\t')
    assert len(indices)==len(np.unique(indices))==4376
    np.testing.assert_array_equal(indices,candidates.raw_index)
    np.testing.assert_array_equal(genes[indices],candidates.Geneid)
    table('candidate_genes.tsv',candidates)
    plan=json.loads((MAPPED/'fold_plan.json').read_text());assert len(plan)==30
    batches=meta.batch.to_numpy(str);coverage={}
    for split in plan:
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        assert len(np.unique(tr))==len(tr) and len(np.unique(te))==len(te)
        assert not np.intersect1d(tr,te).size and set(np.r_[tr,te])==set(range(528))
        assert len(np.unique(y[tr]))==len(np.unique(y[te]))==2
        coverage.setdefault((split['protocol'],split['repeat']),np.zeros(528,int))[te]+=1
        if split['protocol']=='batch_grouped':assert not set(batches[tr])&set(batches[te])
        seen=np.zeros(len(tr),int)
        for inner in split['inner']:
            a,b=np.array(inner['train_positions']),np.array(inner['validation_positions'])
            assert len(np.unique(a))==len(a) and len(np.unique(b))==len(b)
            assert not np.intersect1d(a,b).size and set(np.r_[a,b])==set(range(len(tr)))
            assert len(np.unique(y[tr[a]]))==len(np.unique(y[tr[b]]))==2
            if split['protocol']=='batch_grouped':assert not set(batches[tr[a]])&set(batches[tr[b]])
            seen[b]+=1
        assert (seen==1).all()
    assert len(coverage)==6 and all((x==1).all() for x in coverage.values())
    save('fold_plan.json',plan)
    save('preflight.json',dict(checks=preflight(device),device=str(device),
      device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
      python=sys.executable,sklearn=sklearn.__version__,torch=torch.__version__))
    return raw,genes,meta,indices,plan

def run(device):
    started=time.monotonic();status('PREPARING')
    raw,genes,meta,indices,plan=prepare(device);y=meta.label.to_numpy()
    data=MappedData(raw,genes,y,indices,device);keys=['repeat','protocol','fold']
    old_params=pd.read_csv(MAPPED/'selected_parameters.tsv',sep='\t').query("penalty=='ridge'").set_index(keys)
    old_inner=pd.read_csv(MAPPED/'inner_tuning_scores.tsv',sep='\t').query("penalty=='ridge'").set_index(keys+['C'])
    old_pred=pd.read_csv(MAPPED/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str}).query("penalty=='ridge'")
    old_features=pd.read_csv(MAPPED/'outer_gene_coefficients.tsv',sep='\t').query("penalty=='ridge'")
    features={k:g.Geneid.to_numpy() for k,g in old_features.groupby(keys)}
    predictions=[];metrics=[];parameters=[];tuning=[];weights=[];checks=[];coefficients=[];inner_saved={}
    for number,split in enumerate(plan,1):
        context={k:split[k] for k in keys};key=tuple(context.values())
        status('TRAINING',completed_folds=number-1,total_folds=30,**context)
        print(f'{number}/30 {context}',flush=True)
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        inner=[(np.array(i['train_positions']),np.array(i['validation_positions'])) for i in split['inner']]
        best,rows,pp,ww=tune(data,tr,inner)
        tuning.extend(dict(**context,**r) for r in rows);weights.extend(dict(**context,stage='inner',**r) for r in ww)
        inner_saved[f'{number}_train_indices']=tr
        for (w,c),p in pp.items():inner_saved[f'{number}_{w}_{c}']=p
        for row in rows:
            if row['weight']=='unweighted':
                old=old_inner.loc[key+(row['C'],)]
                np.testing.assert_allclose(row['mean_inner_AUROC'],old.mean_inner_AUROC,atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.fromstring(row['inner_AUROCs'],sep=';'),np.fromstring(old.inner_AUROCs,sep=';'),atol=1e-12,rtol=0)
        assert best['unweighted']['C']==old_params.loc[key].C
        np.testing.assert_allclose(best['unweighted']['threshold'],old_params.loc[key].threshold,atol=1e-8,rtol=0)
        pack=data.pack(tr,te);np.testing.assert_array_equal(genes[pack['state']['indices']],features[key])
        prior=old_pred[(old_pred['repeat']==split['repeat'])&(old_pred.protocol==split['protocol'])&(old_pred.fold==split['fold'])].set_index('PATNO').loc[meta.PATNO.iloc[te]]
        cache={}
        for arm in ARMS:
            chosen=best[arm];candidate=(chosen['weight'],chosen['C'])
            if candidate not in cache:cache[candidate]=fit(pack['train'],y[tr],chosen['C'],chosen['weight'])
            model=cache[candidate];p=model.predict_proba(pack['test'])[:,1]
            if arm=='unweighted':
                np.testing.assert_array_equal(prior.label,y[te])
                np.testing.assert_allclose(p,prior.probability_PD,atol=1e-8,rtol=0)
                checks.append(dict(**context,max_probability_difference=float(np.max(abs(p-prior.probability_PD))),status='PASS'))
            state=artifact(model,pack,chosen,genes,y[tr])
            np.testing.assert_allclose(predict_artifact(state,raw[te[:5]]),p[:5],atol=1e-9,rtol=0)
            class_w=class_weights(y[tr],chosen['weight'])
            parameters.append(dict(**context,**chosen,eligible_genes=pack['train'].shape[1],control_weight=class_w[0],PD_weight=class_w[1],optimizer_iterations=int(model.n_iter_[0])))
            weights.append(dict(**context,stage='outer',arm=arm,weight=chosen['weight'],n_train=len(tr),n_control=int((y[tr]==0).sum()),n_PD=int(y[tr].sum()),control_weight=class_w[0],PD_weight=class_w[1]))
            train_p=model.predict_proba(pack['train'])[:,1]
            metrics.append(dict(**context,arm=arm,n_train=len(tr),n_test=len(te),AUROC=roc_auc_score(y[te],p),AP=average_precision_score(y[te],p),
                Brier=np.mean((p-y[te])**2),log_loss=log_loss(y[te],p),train_AUROC=roc_auc_score(y[tr],train_p),train_AP=average_precision_score(y[tr],train_p)))
            predictions.extend(dict(**context,arm=arm,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(prob),threshold=chosen['threshold']) for i,prob in zip(te,p))
            coefficients.extend(dict(**context,arm=arm,Geneid=g,coefficient=float(c)) for g,c in zip(state['selected_gene_ids'],model.coef_[0]))
        table('out_of_fold_predictions.tsv',predictions);table('outer_fold_metrics.tsv',metrics)
        table('selected_parameters.tsv',parameters);table('inner_tuning_scores.tsv',tuning);table('benchmark_reproduction.tsv',checks)
    pred=pd.DataFrame(predictions)
    assert len(pred)==9504 and not pred.duplicated(['protocol','repeat','arm','PATNO']).any()
    np.savez_compressed(HERE/'inner_oof_probabilities.npz',**inner_saved)
    table('class_weight_audit.tsv',weights);table('outer_coefficients.tsv',coefficients)
    status('FINAL_DEVELOPMENT_FITS',completed_folds=30,total_folds=30)
    final_plan=json.loads((MAPPED/'final_tuning_folds.json').read_text());save('final_tuning_folds.json',final_plan)
    final_splits=[(np.array(s['train_indices']),np.array(s['validation_indices'])) for s in final_plan]
    best,rows,pp,ww=tune(data,np.arange(528),final_splits)
    table('final_tuning_scores.tsv',rows);table('final_inner_class_weights.tsv',ww);save('final_parameters.json',list(best.values()))
    np.savez_compressed(HERE/'final_inner_oof_probabilities.npz',**{f'{w}_{c}':p for (w,c),p in pp.items()})
    pack=data.pack(np.arange(528),np.array([],int));(HERE/'models').mkdir();cache={}
    for arm in ARMS:
        chosen=best[arm];candidate=(chosen['weight'],chosen['C'])
        if candidate not in cache:cache[candidate]=fit(pack['train'],y,chosen['C'],chosen['weight'])
        model=cache[candidate];state=artifact(model,pack,chosen,genes,y)
        state.update(sklearn_version=sklearn.__version__,numpy_version=np.__version__,candidate_gene_ids=genes[indices])
        path=HERE/'models'/f'{arm}_ridge.joblib';joblib.dump(state,path)
        np.testing.assert_allclose(predict_artifact(joblib.load(path),raw[:5]),model.predict_proba(pack['train'][:5])[:,1],atol=1e-9,rtol=0)
    status('SUMMARIZING',completed_folds=30,total_folds=30)
    from report import report
    result=report(HERE,meta,pred,pd.DataFrame(metrics),pd.DataFrame(parameters),device)
    assert all(sha(p)==v for p,v in json.loads((HERE/'input_sha256.json').read_text()).items())
    save('summary.json',dict(status='COMPLETE',validation='PASS',source_hashes_unchanged=True,participants=528,
      device=str(device),reproduced_benchmark_folds=30,main_model_fits=822,elapsed_seconds=time.monotonic()-started,**result))
    status('COMPLETE',completed_folds=30,total_folds=30,elapsed_seconds=time.monotonic()-started)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    # Check before entering the exception handler so a duplicate launch never alters status.
    if '--preflight-only' not in sys.argv and (HERE/'status.json').exists():raise SystemExit('Existing run; refusing overwrite')
    try:
        with threadpool_limits(limits=2):
            device=choose_device()
            if '--require-gpu' in sys.argv and device.type!='cuda':raise RuntimeError('Requested CUDA is unavailable')
            if '--preflight-only' in sys.argv:
                checks=preflight(device);save(f'synthetic_preflight_{device.type}.json',checks);print(checks)
            else:run(device)
    except Exception:
        if '--preflight-only' not in sys.argv:status('FAILED',error=traceback.format_exc())
        traceback.print_exc();sys.exit(1)
