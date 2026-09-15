"""Restartable four-worker DE scheduler followed by matched holdout evaluation."""
from pathlib import Path
import os, sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
for name in ['tmp','cache','matplotlib_cache','models','selectors']:(ROOT/name).mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'matplotlib_cache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import json, time, fcntl, traceback
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from de_workers import run_selector,sha,save,selected
from model_utils import BASE,HALLMARK,Data,choose_device,validate,tune,fit,artifact,predict_artifact,evaluate

def status(stage,**extra):
    save(ROOT/'status.json',dict(stage=stage,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**extra))
    print(stage,extra,flush=True)

def table(name,rows):pd.DataFrame(rows).to_csv(ROOT/name,sep='\t',index=False)

def check_plan(plan,meta):
    train=set(plan['train_indices']);test=set(plan['test_indices'])
    assert len(train)==410 and len(test)==105 and not train&test
    registry={e['name']:e for e in plan['selectors']}
    def selector(name,idx):
        e=registry[name];assert e['train_indices']==sorted(map(int,idx))
        assert e['training_PATNO']==meta.PATNO.iloc[e['train_indices']].tolist()
        assert not set(idx)&test
    for model in plan['models']:
        tr=np.array(model['train_indices']);te=np.array(model['test_indices'])
        assert set(tr)<=train and set(te)<=test and len(set(tr))==len(tr) and len(set(te))==len(te)
        assert set(meta.label.iloc[te])=={0,1}
        seen=[]
        for fold in model['inner']:
            a=tr[fold['train_positions']];b=tr[fold['validation_positions']]
            assert not set(a)&set(b) and set(a)|set(b)==set(tr)
            assert meta.label.iloc[a].value_counts().min()>=2 and set(meta.label.iloc[b])=={0,1}
            selector(fold['selector'],a);seen.extend(b)
        assert sorted(seen)==sorted(tr)
        selector(model['outer_selector'],tr)
    for family in ['combined','sex_only','age_only','pooled_de','pooled_hallmark']:
        indices=[i for m in plan['models'] if m['family']==family for i in m['test_indices']]
        assert sorted(indices)==sorted(test)

def main(workers=4):
    started=time.time();status('PREFLIGHT')
    plan=json.loads((ROOT/'plan.json').read_text())
    for p,digest in json.loads((ROOT/'planning_source_sha256.json').read_text()).items():assert sha(p)==digest,p
    raw=np.load(BASE/'counts.npy',mmap_mode='r');genes=np.load(BASE/'gene_ids.npy')
    meta=pd.read_csv(ROOT/'participant_allocation.tsv',sep='\t',dtype={'PATNO':str})
    assert raw.shape==(528,58780) and meta.PATNO.is_unique and (raw>=0).all()
    assert np.array_equal(meta.label,meta.group.eq('PD').astype(int))
    check_plan(plan,meta)
    device=choose_device();assert device.type=='cuda','Launch with GPU access'
    checks=validate(device);checks['training_and_holdout_boundaries']='PASS'
    save(ROOT/'preflight.json',dict(device=str(device),gpu=torch.cuda.get_device_name(),checks=checks))
    paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.R'))+[ROOT/'plan.json',ROOT/'participant_allocation.tsv',HALLMARK/'membership.npz']
    paths += [BASE/'classifier.py',ROOT.parent/'ppmi-mapped-gene-classifier-2026-09-12/mapped_genes.py',ROOT.parent/'ppmi-classifier-refinement-retry-2026-09-12/refine.py']
    save(ROOT/'run_source_sha256.json',{str(p):sha(p) for p in paths})
    entries=plan['selectors'];completed=[]
    # First fit is the inspected pilot; require it to have completed successfully.
    assert (ROOT/'selectors'/entries[0]['name']/'complete.json').exists(),'Pilot must finish first'
    status('DE_FITTING',completed=0,total=len(entries),workers=workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_selector,e):e for e in entries}
        try:
            for future in as_completed(futures):
                entry=futures[future];result=future.result()
                completed.append(dict(name=entry['name'],**result));table('selector_summary.tsv',completed)
                status('DE_FITTING',completed=len(completed),total=len(entries),last=entry['name'],elapsed_seconds=round(time.time()-started))
        except BaseException:
            for future in futures:future.cancel()
            raise
    status('RIDGE_FITTING',completed=0,total=len(plan['models']))
    y=meta.label.to_numpy();data=Data(raw,genes,y,np.arange(len(genes)),device)
    members=np.load(HALLMARK/'membership.npz')['indices'];assert len(members)==4376
    lookup={g:i for i,g in enumerate(genes)}
    def mask(name):
        t=pd.read_csv(ROOT/'selectors'/name/'selected_genes.tsv',sep='\t')
        return np.array([lookup[g] for g in t.Geneid],int)
    predictions=[];metrics=[];tuning=[];chosen_rows=[]
    for number,m in enumerate(plan['models'],1):
        tr=np.array(m['train_indices']);te=np.array(m['test_indices']);dest=ROOT/'models'/m['name'];dest.mkdir(exist_ok=True)
        hallmark=m['family']=='pooled_hallmark'
        splits=[(np.array(f['train_positions']),np.array(f['validation_positions'])) for f in m['inner']]
        masks=[members if hallmark else mask(f['selector']) for f in m['inner']]
        with threadpool_limits(limits=2):
            best,rows,oof=tune(data,tr,splits,masks)
            pack=data.pack_selected(tr,te,members if hallmark else mask(m['outer_selector']))
            estimator=fit(pack['train'],y[tr],best['C'])
        state=artifact(estimator,pack,best,genes,[])
        state.update(training_n=len(tr),training_PATNO=meta.PATNO.iloc[tr].tolist(),test_PATNO=meta.PATNO.iloc[te].tolist(),model_name=m['name'],family=m['family'])
        if hallmark:state['predictors']='Fixed 4376 Hallmark member universe; training-eligible RNA genes'
        joblib.dump(state,dest/'classifier.joblib')
        p=estimator.predict_proba(pack['test'])[:,1]
        np.testing.assert_allclose(predict_artifact(joblib.load(dest/'classifier.joblib'),raw[te]),p,atol=1e-10,rtol=1e-9)
        pd.DataFrame({'Geneid':state['selected_gene_ids'],'coefficient':estimator.coef_[0],'mean':pack['state']['means'],'scale':pack['state']['scales']}).to_csv(dest/'feature_parameters.tsv',sep='\t',index=False)
        pd.DataFrame({'PATNO':meta.PATNO.iloc[tr].to_numpy(),'label':y[tr],**{'C_'+str(c):v for c,v in oof.items()}}).to_csv(dest/'inner_oof_predictions.tsv',sep='\t',index=False)
        save(dest/'fit_summary.json',dict(**best,training_n=len(tr),testing_n=len(te),features=pack['train'].shape[1],max_iterations=int(np.max(estimator.n_iter_)),intercept_only=pack['train'].shape[1]==0))
        chosen_rows.append(dict(model=m['name'],family=m['family'],**best,training_n=len(tr),test_n=len(te),genes=pack['train'].shape[1]))
        tuning.extend(dict(model=m['name'],**r) for r in rows)
        for j,i in enumerate(te):predictions.append(dict(PATNO=meta.PATNO.iloc[i],global_index=int(i),stratum=meta.stratum.iloc[i],family=m['family'],model=m['name'],label=int(y[i]),probability_PD=float(p[j]),threshold=best['threshold']))
        for mode,threshold in [('fixed_0_5',.5),('inner_selected',best['threshold'])]:metrics.append(dict(model=m['name'],family=m['family'],threshold_mode=mode,**evaluate(y[te],p,threshold)))
        table('holdout_predictions.tsv',predictions);table('model_metrics.tsv',metrics);table('selected_parameters.tsv',chosen_rows);table('inner_tuning_scores.tsv',tuning)
        status('RIDGE_FITTING',completed=number,total=len(plan['models']),last=m['name'])
    status('REPORTING')
    from reporting import report
    report(ROOT,meta,pd.DataFrame(predictions),device)
    for path,digest in json.loads((ROOT/'run_source_sha256.json').read_text()).items():assert sha(path)==digest,path
    for path,digest in json.loads((ROOT/'planning_source_sha256.json').read_text()).items():assert sha(path)==digest,path
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['tmp','cache','matplotlib_cache']) and p.name not in ['run.log','status.json','output_sha256.json','.run.lock']]
    save(ROOT/'output_sha256.json',{str(p):sha(p) for p in paths})
    status('COMPLETE',models=13,DE_fits=len(entries),held_out=105,elapsed_seconds=round(time.time()-started))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--handover',action='store_true')
    args=parser.parse_args();assert 1<=args.workers<=12
    if args.handover:
        import psutil,signal
        previous=json.loads((ROOT/'status.json').read_text())
        try:
            running=psutil.Process(previous['pid'])
            assert any(str(ROOT.name+'/launch.py') in part for part in running.cmdline())
            print('HANDOVER: finishing active fits and cancelling queued jobs',flush=True)
            running.send_signal(signal.SIGINT)
            running.wait(timeout=180)
        except psutil.NoSuchProcess:pass
    lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main(args.workers)
    except BaseException as e:
        status('FAILED',error=repr(e));traceback.print_exc();raise
