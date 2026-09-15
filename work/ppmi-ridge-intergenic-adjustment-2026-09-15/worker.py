from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
import json,time,hashlib,traceback,fcntl
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from adjustment import BASE,MAPPED,HALLMARK,PRIOR,FAMILIES,AdjustData,choose_device,preflight,tune,fit,artifact,predict
BENCH=ROOT.parent/'ppmi-hallmark-blood-block-ridge-2026-09-15'
KEYS=['repeat','protocol','fold']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def table(name,data):pd.DataFrame(data).to_csv(ROOT/name,sep='\t',index=False)
def status(stage,**kw):save('status.json',dict(status=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw));print(stage,kw,flush=True)

def main():
    started=time.time();status('PREFLIGHT');hashes={}
    bi=json.loads((BENCH/'input_sha256.json').read_text());bo=json.loads((BENCH/'output_sha256.json').read_text())
    paths=[BASE/name for name in ['counts.npy','gene_ids.npy','metadata.tsv','classifier.py']]+[MAPPED/name for name in ['mapped_genes.py','fold_plan.json','final_tuning_folds.json','selected_parameters.tsv','inner_tuning_scores.tsv','out_of_fold_predictions.tsv']]
    paths += [HALLMARK/'membership.npz',PRIOR/'refine.py',PRIOR/'elastic_solver.py',BENCH/'primary_summary.tsv',BENCH/'input_sha256.json',BENCH/'output_sha256.json']+list(ROOT.glob('*.py'))+[ROOT/'README.md']
    for p in paths:
        digest=sha(p);expected=bi.get(str(p),bo.get(str(p)))
        if expected is not None:assert digest==expected,str(p)
        hashes[str(p)]=digest
    save('input_sha256.json',hashes)
    raw=np.load(BASE/'counts.npy',mmap_mode='r');genes=np.load(BASE/'gene_ids.npy');meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str});y=meta.label.to_numpy();qc=meta.intergenic_percent.to_numpy(float)
    assert raw.shape==(528,58780) and raw.dtype.kind in 'iu' and (raw>=0).all() and len(genes)==58780
    assert meta.PATNO.is_unique and len(meta)==528 and sum(y)==358 and np.array_equal(y,meta.group.eq('PD').astype(int))
    indices=np.load(HALLMARK/'membership.npz')['indices'];assert len(indices)==4376 and len(np.unique(indices))==4376
    plan=json.loads((MAPPED/'fold_plan.json').read_text());assert len(plan)==30
    cover={}
    for s in plan:
        tr=np.array(s['train_indices']);te=np.array(s['test_indices']);assert len(set(tr))==len(tr) and len(set(te))==len(te) and not set(tr)&set(te) and set(tr)|set(te)==set(range(528))
        assert set(y[tr])==set(y[te])=={0,1}
        cover.setdefault((s['protocol'],s['repeat']),np.zeros(528,int))[te]+=1
        if s['protocol']=='batch_grouped':assert not set(meta.batch.iloc[tr])&set(meta.batch.iloc[te])
        seen=np.zeros(len(tr),int)
        for f in s['inner']:
            a=np.array(f['train_positions']);b=np.array(f['validation_positions']);assert not set(a)&set(b) and set(a)|set(b)==set(range(len(tr)))
            assert set(y[tr[a]])==set(y[tr[b]])=={0,1};seen[b]+=1
            if s['protocol']=='batch_grouped':assert not set(meta.batch.iloc[tr[a]])&set(meta.batch.iloc[tr[b]])
        assert (seen==1).all()
    assert len(cover)==6 and all((v==1).all() for v in cover.values());save('fold_plan.json',plan)
    device=choose_device();assert device.type=='cuda','RTX required';torch.set_num_threads(2)
    check=preflight(device);check['gpu']=torch.cuda.get_device_name(device);save('preflight.json',check)
    data=AdjustData(raw,genes,y,indices,qc,device)
    oldparams=pd.read_csv(MAPPED/'selected_parameters.tsv',sep='\t').query("penalty=='ridge'").set_index(KEYS)
    oldinner=pd.read_csv(MAPPED/'inner_tuning_scores.tsv',sep='\t').query("penalty=='ridge'").set_index(KEYS+['C'])
    oldpred=pd.read_csv(MAPPED/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str}).query("penalty=='ridge'")
    results=[];metrics=[];parameters=[];tuning=[];checks=[];qc_flags=[]
    for number,s in enumerate(plan,1):
        context={k:s[k] for k in KEYS};key=tuple(context[k] for k in KEYS);status('TRAINING',completed_folds=number-1,total_folds=30,**context)
        tr=np.array(s['train_indices']);te=np.array(s['test_indices']);inner=[(np.array(f['train_positions']),np.array(f['validation_positions'])) for f in s['inner']]
        best,rows,innerp=tune(data,tr,inner);tuning.extend(dict(**context,**r) for r in rows)
        for r in rows:
            if r['family']=='rna':
                ref=oldinner.loc[key+(r['C'],)];np.testing.assert_allclose(r['mean_inner_AUROC'],ref.mean_inner_AUROC,atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.fromstring(r['inner_AUROCs'],sep=';'),np.fromstring(ref.inner_AUROCs,sep=';'),atol=1e-12,rtol=0)
        ref=oldparams.loc[key];assert best['rna']['C']==ref.C;np.testing.assert_allclose(best['rna']['threshold'],ref.threshold,atol=1e-8,rtol=0)
        dest=ROOT/'folds'/f'fold_{number:02d}';dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'inner_predictions.npz',train_indices=tr,**{f'{f}_{c}':v for (f,c),v in innerp.items()})
        packs=data.pack(tr,te)
        previous=oldpred[(oldpred.protocol==s['protocol'])&(oldpred.repeat==s['repeat'])&(oldpred.fold==s['fold'])].set_index('PATNO').loc[meta.PATNO.iloc[te]]
        assert np.array_equal(previous.label,y[te])
        for f in FAMILIES:
            pack=packs[f];chosen=best[f];model=fit(pack['train'],y[tr],chosen['C']);p=model.predict_proba(pack['test'])[:,1];tp=model.predict_proba(pack['train'])[:,1]
            if f=='rna':np.testing.assert_allclose(p,previous.probability_PD,atol=1e-8,rtol=1e-8)
            state=artifact(model,pack,chosen,genes);joblib.dump(state,dest/(f+'.joblib'))
            ip=predict(joblib.load(dest/(f+'.joblib')),raw[te],qc[te]);np.testing.assert_allclose(ip,p,atol=1e-8,rtol=1e-8)
            checks.append(dict(**context,family=f,inference_max_error=float(np.max(abs(ip-p))),benchmark_max_error=float(np.max(abs(p-previous.probability_PD))) if f=='rna' else None,status='PASS'))
            results.extend(dict(**context,family=f,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(v),threshold=chosen['threshold']) for i,v in zip(te,p))
            metrics.append(dict(**context,family=f,n_train=len(tr),n_test=len(te),n_features=pack['train'].shape[1],AUROC=roc_auc_score(y[te],p),AP=average_precision_score(y[te],p),train_AUROC=roc_auc_score(y[tr],tp),train_AP=average_precision_score(y[tr],tp)))
            parameters.append(dict(**context,**chosen,n_features=pack['train'].shape[1],optimizer_iterations=int(model.n_iter_[0])))
        a=packs['intergenic_adjusted']['adjustment'];names=genes[packs['rna']['state']['indices']]
        pd.DataFrame(dict(Geneid=names,technical_slope_standardized=a['beta'],residual_SD=a['residual_scales'],retained=a['residual_keep'])).to_csv(dest/'gene_adjustment.tsv.gz',sep='\t',index=False)
        qc_flags.extend(dict(**context,PATNO=meta.PATNO.iloc[i],intergenic_percent=float(qc[i]),outside_training_range=bool(qc[i]<a['training_qc_min'] or qc[i]>a['training_qc_max']),training_min=a['training_qc_min'],training_max=a['training_qc_max']) for i in te)
        table('out_of_fold_predictions.tsv',results);table('outer_fold_metrics.tsv',metrics);table('selected_parameters.tsv',parameters);table('inner_tuning_scores.tsv',tuning);table('fold_validation.tsv',checks)
    pred=pd.DataFrame(results);assert len(pred)==6336 and not pred.duplicated(['protocol','repeat','family','PATNO']).any();table('heldout_technical_range.tsv',qc_flags)
    status('FINAL_DEVELOPMENT_FITS',completed_folds=30,total_folds=30)
    final=json.loads((MAPPED/'final_tuning_folds.json').read_text());save('final_tuning_folds.json',final);splits=[(np.array(f['train_indices']),np.array(f['validation_indices'])) for f in final]
    seen=np.zeros(528,int)
    for a,b in splits:
        assert not set(a)&set(b) and set(a)|set(b)==set(range(528));seen[b]+=1
    assert (seen==1).all()
    best,rows,ip=tune(data,np.arange(528),splits);table('final_tuning_scores.tsv',rows);save('final_parameters.json',list(best.values()))
    np.savez_compressed(ROOT/'final_inner_predictions.npz',**{f'{f}_{c}':v for (f,c),v in ip.items()})
    packs=data.pack(np.arange(528),np.array([],int));(ROOT/'models').mkdir(exist_ok=True)
    for f in FAMILIES:
        model=fit(packs[f]['train'],y,best[f]['C']);s=artifact(model,packs[f],best[f],genes);s.update(training_n=528,training_PATNO=meta.PATNO.tolist());file=ROOT/'models'/(f+'_ridge.joblib');joblib.dump(s,file)
        np.testing.assert_allclose(predict(joblib.load(file),raw[:5],qc[:5]),model.predict_proba(packs[f]['train'][:5])[:,1],atol=1e-8,rtol=1e-8)
    status('REPORTING');from reporting import report
    result=report(ROOT,meta,pred,pd.DataFrame(metrics),device)
    prior=pd.read_csv(BENCH/'primary_summary.tsv',sep='\t').query("family=='rna'").set_index('protocol');now=pd.read_csv(ROOT/'primary_summary.tsv',sep='\t').query("family=='rna'").set_index('protocol')
    cols=['AUROC_mean','AUROC_ci_lower','AUROC_ci_upper','AP_mean','AP_ci_lower','AP_ci_upper']
    np.testing.assert_allclose(prior[cols].sort_index(),now[cols].sort_index(),atol=1e-12,rtol=0)
    for p,h in hashes.items():assert sha(p)==h,p
    result.update(status='COMPLETE',participants=528,source_hashes_unchanged=True);save('summary.json',result)
    save('validation.json',dict(status='PASS',participants=528,original_folds_reproduced=30,outer_inference_checks=60,final_artifact_checks=2,baseline_primary_intervals_reproduced=True,source_hashes_unchanged=len(hashes)))
    save('output_sha256.json',{str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and 'matplotlib_cache' not in p.parts and p.name not in ['run.log','status.json','output_sha256.json','.run.lock','.launch.lock']})
    status('COMPLETE',completed_folds=30,total_folds=30,elapsed_seconds=round(time.time()-started,2))

if __name__=='__main__':
    lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with threadpool_limits(limits=2):main()
    except BaseException as e:status('FAILED',error=repr(e));traceback.print_exc();raise
