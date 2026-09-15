"""Three annotation-only score classifiers on original repeated nested folds."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;BENCH=ROOT.parent/'ppmi-hallmark-blood-block-ridge-2026-09-15'
for d in ['tmp','cache','matplotlib_cache','folds','models']:(ROOT/d).mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'matplotlib_cache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import json,time,hashlib,traceback,shutil,fcntl
import numpy as np
import pandas as pd
import torch,joblib,sklearn
from scipy import sparse
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score,average_precision_score
from path_models import BASE,MAPPED,PATH_FAMILIES,FAMILIES,PathData,choose_device,tune,fit,artifact,predict_artifact
from validation import preflight
KEYS=['repeat','protocol','fold']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)

def table(name,rows):pd.DataFrame(rows).to_csv(ROOT/name,sep='\t',index=False)

def status(stage,**extra):
    save('status.json',dict(status=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra));print(stage,extra,flush=True)

def check_plan(plan,meta):
    coverage={};y=meta.label.to_numpy();batches=meta.batch.to_numpy(str)
    for s in plan:
        tr=np.array(s['train_indices']);te=np.array(s['test_indices'])
        assert len(set(tr))==len(tr) and len(set(te))==len(te) and not set(tr)&set(te) and set(tr)|set(te)==set(range(528))
        assert set(y[tr])==set(y[te])=={0,1}
        coverage.setdefault((s['protocol'],s['repeat']),np.zeros(528,int))[te]+=1
        if s['protocol']=='batch_grouped':assert not set(batches[tr])&set(batches[te])
        seen=np.zeros(len(tr),int)
        for f in s['inner']:
            a=np.array(f['train_positions']);b=np.array(f['validation_positions']);assert not set(a)&set(b) and set(a)|set(b)==set(range(len(tr)))
            assert set(y[tr[a]])==set(y[tr[b]])=={0,1};seen[b]+=1
            if s['protocol']=='batch_grouped':assert not set(batches[tr[a]])&set(batches[tr[b]])
        assert (seen==1).all()
    assert len(coverage)==6 and all((v==1).all() for v in coverage.values())

def main():
    start=time.time();status('PREFLIGHT');hashes={}
    assert json.loads((BENCH/'status.json').read_text())['status']=='COMPLETE'
    baseline=json.loads((BENCH/'output_sha256.json').read_text());baseinputs=json.loads((BENCH/'input_sha256.json').read_text())
    def consume(p,expected=None):
        p=Path(p);digest=sha(p)
        if expected is not None:assert digest==expected,str(p)
        hashes[str(p)]=digest;return p
    for p,digest in json.loads((ROOT/'mapping_input_sha256.json').read_text()).items():consume(p,digest)
    for name in ['counts.npy','metadata.tsv','gene_ids.npy','classifier.py']:consume(BASE/name,baseinputs[str(BASE/name)])
    for name in ['fold_plan.json','final_tuning_folds.json','out_of_fold_predictions.tsv','outer_fold_metrics.tsv','primary_summary.tsv']:
        p=consume(BENCH/name,baseline[str(BENCH/name)]);shutil.copyfile(p,ROOT/('benchmark_'+name if name not in ['fold_plan.json','final_tuning_folds.json'] else name))
    for p in [BENCH/'output_sha256.json',BENCH/'input_sha256.json',MAPPED/'out_of_fold_predictions.tsv',MAPPED/'fold_plan.json',MAPPED/'mapped_genes.py',ROOT.parent/'ppmi-classifier-refinement-retry-2026-09-12/refine.py',ROOT.parent/'ppmi-classifier-refinement-retry-2026-09-12/elastic_solver.py']:
        consume(p,baseinputs.get(str(p)))
    for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.R'))+[ROOT/x for x in ['README.md','gene_reference.tsv','pathway_reference.tsv','membership.npz','mapping_summary.json','mapping_input_sha256.json']]:consume(p)
    save('input_sha256.json',hashes)
    raw=np.load(BASE/'counts.npy',mmap_mode='r');genes=np.load(BASE/'gene_ids.npy');meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str});y=meta.label.to_numpy()
    assert raw.shape==(528,58780) and meta.PATNO.is_unique and meta.group.value_counts().to_dict()=={'PD':358,'Control':170} and np.array_equal(y,meta.group.eq('PD').astype(int))
    assert raw.dtype.kind in 'iu' and (raw>=0).all();table('metadata.tsv',meta)
    gene_ref=pd.read_csv(ROOT/'gene_reference.tsv',sep='\t');np.testing.assert_array_equal(gene_ref.Geneid,genes)
    reference=pd.read_csv(ROOT/'pathway_reference.tsv',sep='\t');membership=sparse.load_npz(ROOT/'membership.npz')
    assert membership.shape==(len(reference),len(genes)) and membership.max()==1
    plan=json.loads((ROOT/'fold_plan.json').read_text());assert plan==json.loads((MAPPED/'fold_plan.json').read_text());check_plan(plan,meta)
    oldp=pd.read_csv(ROOT/'benchmark_out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str});oldm=pd.read_csv(ROOT/'benchmark_outer_fold_metrics.tsv',sep='\t')
    predictions=oldp.loc[oldp.family.isin(['rna','rna_blood'])].to_dict('records');metrics=oldm.loc[oldm.family.isin(['rna','rna_blood'])].to_dict('records')
    mapped=pd.read_csv(MAPPED/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str});mapped=mapped.loc[mapped.penalty.eq('ridge')].set_index(KEYS+['PATNO']).sort_index()
    actual=oldp.loc[oldp.family.eq('rna')].set_index(KEYS+['PATNO']).sort_index()
    assert actual.index.equals(mapped.index);np.testing.assert_allclose(actual.probability_PD,mapped.probability_PD,atol=1e-8,rtol=1e-8)
    for s in plan:
        for f in ['rna','rna_blood']:
            part=oldp.loc[(oldp.family==f)&(oldp.protocol==s['protocol'])&(oldp.repeat==s['repeat'])&(oldp.fold==s['fold'])].set_index('PATNO').loc[meta.PATNO.iloc[s['test_indices']]]
            assert len(part)==len(s['test_indices']) and np.array_equal(part.label,y[s['test_indices']])
    device=choose_device();assert device.type=='cuda','NVIDIA GPU required';torch.set_num_threads(2)
    checks=preflight(device);save('preflight.json',dict(status='PASS',checks=checks,gpu=torch.cuda.get_device_name(),sklearn=sklearn.__version__,torch=torch.__version__,python=sys.version,benchmark_folds_verified=60))
    data=PathData(raw,genes,y,gene_ref.Length.to_numpy(float),gene_ref.unambiguous.to_numpy(bool),membership,reference,device)
    params=[];tuning=[];coefficients=[];checks=[]
    for number,s in enumerate(plan,1):
        context={k:s[k] for k in KEYS};status('TRAINING',completed_folds=number-1,total_folds=30,**context)
        tr=np.array(s['train_indices']);te=np.array(s['test_indices']);splits=[(np.array(f['train_positions']),np.array(f['validation_positions'])) for f in s['inner']]
        best,rows,oof,coverage=tune(data,tr,splits);tuning.extend(dict(**context,**r) for r in rows)
        dest=ROOT/'folds'/f'fold_{number:02d}';dest.mkdir(exist_ok=True)
        coverage.to_csv(dest/'inner_pathway_coverage.tsv.gz',sep='\t',index=False)
        np.savez_compressed(dest/'inner_oof_predictions.npz',train_indices=tr,**{f'{f}_{c}':v for (f,c),v in oof.items()})
        packs,coverage=data.pack(tr,te);coverage.to_csv(dest/'outer_pathway_coverage.tsv.gz',sep='\t',index=False)
        np.save(dest/'training_background_indices.npy',packs[PATH_FAMILIES[0]]['state']['background_indices'])
        for family in PATH_FAMILIES:
            pack=packs[family];chosen=best[family];model=fit(pack['train'],y[tr],chosen['C']);p=model.predict_proba(pack['test'])[:,1];trainp=model.predict_proba(pack['train'])[:,1]
            state=artifact(model,pack,chosen,genes);inferred=predict_artifact(state,raw[te]);np.testing.assert_allclose(inferred,p,atol=1e-8,rtol=1e-8)
            checks.append(dict(**context,family=family,maximum_inference_error=float(np.max(abs(inferred-p))),status='PASS'))
            metrics.append(dict(**context,family=family,n_train=len(tr),n_test=len(te),n_features=pack['train'].shape[1],AUROC=float(roc_auc_score(y[te],p)),AP=float(average_precision_score(y[te],p)),train_AUROC=float(roc_auc_score(y[tr],trainp)),train_AP=float(average_precision_score(y[tr],trainp))))
            predictions.extend(dict(**context,family=family,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(prob),threshold=chosen['threshold']) for i,prob in zip(te,p))
            params.append(dict(**context,**chosen,n_features=pack['train'].shape[1],optimizer_iterations=int(model.n_iter_[0])))
            coefficients.extend(dict(**context,family=family,pathway=name,coefficient=float(c)) for name,c in zip(pack['state']['pathway_names'],model.coef_[0]))
            pd.DataFrame(dict(pathway=pack['state']['pathway_names'],means=pack['state']['means'],scales=pack['state']['scales'])).to_csv(dest/(family+'_parameters.tsv'),sep='\t',index=False)
        table('out_of_fold_predictions.tsv',predictions);table('outer_fold_metrics.tsv',metrics);table('selected_parameters.tsv',params);table('inner_tuning_scores.tsv',tuning);table('inference_validation.tsv',checks)
    pred=pd.DataFrame(predictions);assert len(pred)==528*3*2*5 and not pred.duplicated(['repeat','protocol','family','PATNO']).any()
    table('outer_pathway_coefficients.tsv',coefficients);status('FINAL_DEVELOPMENT_FITS',completed_folds=30,total_folds=30)
    final=json.loads((ROOT/'final_tuning_folds.json').read_text());splits=[(np.array(f['train_indices']),np.array(f['validation_indices'])) for f in final]
    seen=np.zeros(528,int)
    for a,b in splits:
        assert not set(a)&set(b) and set(a)|set(b)==set(range(528));seen[b]+=1
    assert (seen==1).all()
    best,rows,oof,coverage=tune(data,np.arange(528),splits);table('final_tuning_scores.tsv',rows);coverage.to_csv(ROOT/'final_tuning_pathway_coverage.tsv.gz',sep='\t',index=False)
    np.savez_compressed(ROOT/'final_inner_oof_predictions.npz',**{f'{f}_{c}':v for (f,c),v in oof.items()})
    packs,coverage=data.pack(np.arange(528),np.array([],int));coverage.to_csv(ROOT/'final_pathway_coverage.tsv',sep='\t',index=False)
    for family in PATH_FAMILIES:
        pack=packs[family];model=fit(pack['train'],y,best[family]['C']);s=artifact(model,pack,best[family],genes);s.update(training_n=528,training_PATNO=meta.PATNO.tolist(),sklearn_version=sklearn.__version__)
        p=ROOT/'models'/(family+'_ridge.joblib');joblib.dump(s,p);np.testing.assert_allclose(predict_artifact(joblib.load(p),raw[:5]),model.predict_proba(pack['train'][:5])[:,1],atol=1e-8,rtol=1e-8)
    save('final_parameters.json',list(best.values()));status('REPORTING')
    from reporting import report
    result=report(ROOT,meta,pred,pd.DataFrame(metrics),device)
    old=pd.read_csv(ROOT/'benchmark_primary_summary.tsv',sep='\t').set_index(['protocol','family']);now=pd.read_csv(ROOT/'primary_summary.tsv',sep='\t').set_index(['protocol','family'])
    cols=['AUROC_mean','AUROC_ci_lower','AUROC_ci_upper','AP_mean','AP_ci_lower','AP_ci_upper']
    for p in ['participant_stratified','batch_grouped']:
        for f in ['rna','rna_blood']:np.testing.assert_allclose(old.loc[(p,f),cols].to_numpy(float),now.loc[(p,f),cols].to_numpy(float),atol=1e-12,rtol=0)
    for path,digest in hashes.items():assert sha(path)==digest,path
    result.update(status='COMPLETE',participants=528,benchmark_folds_verified=60,outer_inference_checks=90,final_artifacts=3,source_hashes_unchanged=True,elapsed_seconds=time.time()-start)
    save('summary.json',result);save('final_audit.json',dict(status='PASS',benchmark_primary_intervals_reproduced=4,outer_inference_checks=90,final_artifact_roundtrips=3,source_hashes_unchanged=len(hashes)))
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['tmp','cache','matplotlib_cache']) and p.name not in ['status.json','run.log','output_sha256.json','.launch.lock','.run.lock']]
    save('output_sha256.json',{str(p):sha(p) for p in files});status('COMPLETE',completed_folds=30,total_folds=30,elapsed_seconds=round(time.time()-start))

if __name__=='__main__':
    lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with threadpool_limits(limits=2):main()
    except BaseException as e:status('FAILED',error=repr(e));traceback.print_exc();raise
