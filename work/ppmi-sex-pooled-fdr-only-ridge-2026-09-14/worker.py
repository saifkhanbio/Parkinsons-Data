"""FDR-only masks from verified cached training DE, matched ridge assessment."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'ppmi-sex-age-stratified-de-ridge-2026-09-14'
for name in ['tmp','cache','matplotlib_cache','models','selectors']:(ROOT/name).mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'matplotlib_cache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
import json,time,hashlib,shutil,traceback,fcntl
import numpy as np
import pandas as pd
import joblib,torch,sklearn
from threadpoolctl import threadpool_limits
from model_utils import BASE,Data,choose_device,fit,tune,validate,artifact,predict_artifact,evaluate
from plan_checks import check_plan

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)

def table(name,rows):pd.DataFrame(rows).to_csv(ROOT/name,sep='\t',index=False)

def status(stage,**extra):
    save('status.json',dict(stage=stage,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra));print(stage,extra,flush=True)

def select(frame):
    finite=np.isfinite(frame[['stat','pvalue','padj','log2FoldChange']].to_numpy(float)).all(1)
    return frame.loc[finite & frame.beta_converged.eq(True) & frame.padj.lt(.05)].copy()

def selection_test():
    t=pd.DataFrame(dict(Geneid=['small','zero','boundary','missing','failed'],stat=[2.]*5,pvalue=[.01]*5,padj=[.049,.01,.05,np.nan,.001],log2FoldChange=[.01,0.,1.,1.,1.],beta_converged=[True,True,True,True,False]))
    assert select(t).Geneid.tolist()==['small','zero']

def main():
    start=time.time();status('PREFLIGHT');selection_test()
    assert json.loads((SOURCE/'status.json').read_text())['stage']=='COMPLETE'
    ref=json.loads((SOURCE/'output_sha256.json').read_text());up=json.loads((SOURCE/'planning_source_sha256.json').read_text());source_code=json.loads((SOURCE/'run_source_sha256.json').read_text());hashes={}
    def consume(p,expected=None):
        p=Path(p);digest=sha(p)
        if expected is not None:assert digest==expected,str(p)
        hashes[str(p)]=digest;return p
    def source(name):
        p=SOURCE/name;return consume(p,ref[str(p)])
    for name in ['plan.json','participant_allocation.tsv','bootstrap_weights.npz','holdout_predictions.tsv','heldout_performance.tsv']:
        shutil.copyfile(source(name),ROOT/('prior_'+name if name in ['holdout_predictions.tsv','heldout_performance.tsv'] else name))
    for name in ['counts.npy','metadata.tsv','gene_ids.npy']:consume(BASE/name,up[str(BASE/name)])
    for name in ['output_sha256.json','planning_source_sha256.json','run_source_sha256.json']:consume(SOURCE/name)
    for folder,name in [('ppmi-classifier-2026-09-12','classifier.py'),('ppmi-mapped-gene-classifier-2026-09-12','mapped_genes.py'),('ppmi-classifier-refinement-retry-2026-09-12','refine.py'),('ppmi-classifier-refinement-retry-2026-09-12','elastic_solver.py')]:
        p=ROOT.parent/folder/name;consume(p,source_code.get(str(p)))
    for p in list(ROOT.glob('*.py'))+[ROOT/'README.md']:consume(p)
    meta=pd.read_csv(ROOT/'participant_allocation.tsv',sep='\t',dtype={'PATNO':str});plan=json.loads((ROOT/'plan.json').read_text());check_plan(plan,meta)
    raw=np.load(BASE/'counts.npy',mmap_mode='r');genes=np.load(BASE/'gene_ids.npy');y=meta.label.to_numpy()
    assert raw.shape==(528,58780) and meta.PATNO.is_unique and (raw>=0).all() and raw.dtype.kind in 'iu'
    assert np.array_equal(y,meta.group.eq('PD').astype(int));lookup={g:i for i,g in enumerate(genes)}
    models=[m for m in plan['models'] if m['name'] in ['sex_Female','sex_Male','pooled_de']];assert len(models)==3
    needed={m['outer_selector'] for m in models}|{f['selector'] for m in models for f in m['inner']};assert len(needed)==33
    masks={};selection_rows=[]
    for e in plan['selectors']:
        if e['name'] not in needed:continue
        d=Path('selectors')/e['name'];assert json.loads(source(d/'input.json').read_text())==e
        t=pd.read_csv(source(d/'de_results.tsv.gz'),sep='\t');strict=pd.read_csv(source(d/'selected_genes.tsv'),sep='\t');s=select(t)
        assert s.Geneid.is_unique and set(s.Geneid)<=set(genes) and set(strict.Geneid)<=set(s.Geneid)
        assert s.loc[s.log2FoldChange.abs().gt(.5),'Geneid'].tolist()==strict.Geneid.tolist()
        dest=ROOT/d;dest.mkdir(exist_ok=True);s.to_csv(dest/'selected_genes.tsv',sep='\t',index=False)
        save(str(d/'training_membership.json'),e)
        masks[e['name']]=np.array([lookup[g] for g in s.Geneid],int)
        selection_rows.append(dict(selector=e['name'],training_n=len(e['train_indices']),strict_genes=len(strict),FDR_only_genes=len(s),added_genes=len(s)-len(strict)))
    table('selector_summary.tsv',selection_rows)
    prior=pd.read_csv(ROOT/'prior_holdout_predictions.tsv',sep='\t',dtype={'PATNO':str})
    for name in ['sex_Female','sex_Male','pooled_de','pooled_hallmark']:source(Path('models')/name/'classifier.joblib')
    save('input_sha256.json',hashes)
    device=choose_device();assert device.type=='cuda','NVIDIA GPU access required'
    checks=validate(device);save('preflight.json',dict(checks=checks,selection_cutoffs='PASS',verified_DE_masks=33,gpu=torch.cuda.get_device_name(),python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,torch=torch.__version__))
    data=Data(raw,genes,y,np.arange(len(genes)),device)
    # Reproduce the four reference artifacts on their exact original held-out people.
    for name in ['sex_Female','sex_Male','pooled_de','pooled_hallmark']:
        m=next(m for m in plan['models'] if m['name']==name);te=np.array(m['test_indices'])
        old=joblib.load(SOURCE/'models'/name/'classifier.joblib');assert old['training_PATNO']==meta.PATNO.iloc[m['train_indices']].tolist()
        rows=prior.loc[prior.model.eq(name)].set_index('PATNO').loc[meta.PATNO.iloc[te]]
        np.testing.assert_allclose(predict_artifact(old,raw[te]),rows.probability_PD,atol=1e-10,rtol=0)
    predictions=[];parameters=[];tuning=[];metrics=[]
    for number,m in enumerate(models,1):
        status('RIDGE_FITTING',model=m['name'],completed=number-1,total=3)
        tr=np.array(m['train_indices']);te=np.array(m['test_indices']);splits=[(np.array(f['train_positions']),np.array(f['validation_positions'])) for f in m['inner']]
        with threadpool_limits(limits=2):
            best,rows,oof=tune(data,tr,splits,[masks[f['selector']] for f in m['inner']])
            pack=data.pack_selected(tr,te,masks[m['outer_selector']]);model=fit(pack['train'],y[tr],best['C'])
        state=artifact(model,pack,best,genes,[]);state.update(model_name=m['name'],family=m['family'],training_n=len(tr),training_PATNO=meta.PATNO.iloc[tr].tolist(),test_PATNO=meta.PATNO.iloc[te].tolist())
        dest=ROOT/'models'/m['name'];dest.mkdir(exist_ok=True);joblib.dump(state,dest/'classifier.joblib')
        p=model.predict_proba(pack['test'])[:,1];np.testing.assert_allclose(predict_artifact(joblib.load(dest/'classifier.joblib'),raw[te]),p,atol=1e-10,rtol=0)
        pd.DataFrame(dict(Geneid=state['selected_gene_ids'],coefficient=model.coef_[0],mean=pack['state']['means'],scale=pack['state']['scales'])).to_csv(dest/'feature_parameters.tsv',sep='\t',index=False)
        pd.DataFrame({'PATNO':meta.PATNO.iloc[tr].to_numpy(),'label':y[tr],**{'C_'+str(c):v for c,v in oof.items()}}).to_csv(dest/'inner_oof_predictions.tsv',sep='\t',index=False)
        info=dict(model=m['name'],family=m['family'],**best,training_n=len(tr),test_n=len(te),genes=pack['train'].shape[1],intercept_only=pack['train'].shape[1]==0)
        save(str(Path('models')/m['name']/'fit_summary.json'),info);parameters.append(info);tuning.extend(dict(model=m['name'],**r) for r in rows)
        for j,i in enumerate(te):predictions.append(dict(PATNO=meta.PATNO.iloc[i],global_index=int(i),family=m['family'],model=m['name'],label=int(y[i]),probability_PD=float(p[j]),threshold=best['threshold']))
        for mode,t in [('fixed_0_5',.5),('inner_selected',best['threshold'])]:metrics.append(dict(model=m['name'],threshold_mode=mode,**evaluate(y[te],p,t)))
        table('holdout_predictions.tsv',predictions);table('selected_parameters.tsv',parameters);table('inner_tuning_scores.tsv',tuning);table('model_metrics.tsv',metrics)
    status('REPORTING')
    from reporting import report
    report(ROOT,meta,pd.DataFrame(predictions),prior,device)
    for p,digest in hashes.items():assert sha(p)==digest,p
    save('final_audit.json',dict(status='PASS',training_DE_masks=33,new_models=3,reference_artifacts_reproduced=4,new_inference_roundtrips=3,source_hashes_unchanged=len(hashes)))
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['tmp','cache','matplotlib_cache']) and p.name not in ['run.log','status.json','output_sha256.json','.run.lock','.launch.lock']]
    save('output_sha256.json',{str(p):sha(p) for p in files});status('COMPLETE',models=3,heldout_participants=105,elapsed_seconds=round(time.time()-start))

if __name__=='__main__':
    lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:status('FAILED',error=repr(e));traceback.print_exc();raise
