"""Verified cached-DE random forests on unchanged demographic holdouts."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'ppmi-sex-age-stratified-de-ridge-2026-09-14'
for name in ['tmp','cache','matplotlib_cache','models']:(ROOT/name).mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'matplotlib_cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import json,time,hashlib,fcntl,traceback,shutil
import numpy as np
import pandas as pd
import joblib,torch
from model_utils import BASE,HALLMARK,Data,choose_device,artifact,predict_artifact,evaluate
from forest import GRID,fit,tune,validate
from plan_checks import check_plan

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=ROOT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def table(name,rows):pd.DataFrame(rows).to_csv(ROOT/name,sep='\t',index=False)

def status(stage,**extra):
    save('status.json',dict(stage=stage,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))
    print(stage,extra,flush=True)

def main():
    start=time.time();status('PREFLIGHT')
    assert json.loads((SOURCE/'status.json').read_text())['stage']=='COMPLETE'
    reference=json.loads((SOURCE/'output_sha256.json').read_text())
    upstream=json.loads((SOURCE/'planning_source_sha256.json').read_text())
    hashes={}
    def consume(p,expected=None):
        p=Path(p);digest=sha(p)
        if expected is not None:assert digest==expected,str(p)
        hashes[str(p)]=digest
    def source(name):
        p=SOURCE/name;consume(p,reference[str(p)]);return p
    for name in ['plan.json','participant_allocation.tsv','bootstrap_weights.npz','holdout_predictions.tsv','selected_parameters.tsv','heldout_performance.tsv']:
        p=source(name);shutil.copyfile(p,ROOT/('ridge_'+name if name in ['holdout_predictions.tsv','selected_parameters.tsv','heldout_performance.tsv'] else name))
    consume(SOURCE/'output_sha256.json');consume(SOURCE/'planning_source_sha256.json')
    for name in ['counts.npy','gene_ids.npy','metadata.tsv']:consume(BASE/name,upstream[str(BASE/name)])
    consume(HALLMARK/'membership.npz',json.loads((SOURCE/'run_source_sha256.json').read_text())[str(HALLMARK/'membership.npz')])
    for p in list(ROOT.glob('*.py'))+[ROOT/'README.md']:consume(p)
    plan=json.loads((ROOT/'plan.json').read_text());meta=pd.read_csv(ROOT/'participant_allocation.tsv',sep='\t',dtype={'PATNO':str})
    raw=np.load(BASE/'counts.npy',mmap_mode='r');genes=np.load(BASE/'gene_ids.npy');y=meta.label.to_numpy()
    assert raw.shape==(528,58780) and len(meta)==528 and meta.PATNO.is_unique and np.array_equal(y,meta.group.eq('PD').astype(int))
    assert (raw>=0).all() and raw.dtype.kind in 'iu'
    check_plan(plan,meta);lookup={g:i for i,g in enumerate(genes)};masks={}
    for e in plan['selectors']:
        d=Path('selectors')/e['name']
        inp=json.loads(source(d/'input.json').read_text());assert inp==e
        source(d/'complete.json');frame=pd.read_csv(source(d/'de_results.tsv.gz'),sep='\t')
        chosen=pd.read_csv(source(d/'selected_genes.tsv'),sep='\t')
        finite=np.isfinite(frame[['stat','pvalue','padj','log2FoldChange']].to_numpy(float)).all(1)
        selected=frame.loc[finite & frame.beta_converged.eq(True) & frame.padj.lt(.05) & frame.log2FoldChange.abs().gt(.5)]
        assert selected.Geneid.tolist()==chosen.Geneid.tolist()
        assert chosen.Geneid.is_unique and set(chosen.Geneid)<=set(genes)
        masks[e['name']]=np.array([lookup[g] for g in chosen.Geneid],int)
    members=np.load(HALLMARK/'membership.npz')['indices'];assert len(members)==4376
    ridge_pred=pd.read_csv(ROOT/'ridge_holdout_predictions.tsv',sep='\t',dtype={'PATNO':str})
    for m in plan['models']:source(Path('models')/m['name']/'classifier.joblib')
    save('input_sha256.json',hashes);save('candidate_grid.json',GRID)
    device=choose_device();assert device.type=='cuda','Launch with NVIDIA access'
    checks=validate(device);save('preflight.json',dict(gpu=torch.cuda.get_device_name(),checks=checks,selector_masks_checked=len(masks),plan_boundaries='PASS'))
    data=Data(raw,genes,y,np.arange(len(genes)),device)
    predictions=[];chosen_rows=[];scores=[];metrics=[];roundtrips=[]
    for number,m in enumerate(plan['models'],1):
        name=m['name'];status('RF_TUNING',model=name,completed_models=number-1,total_models=13)
        tr=np.array(m['train_indices']);te=np.array(m['test_indices']);hallmark=m['family']=='pooled_hallmark'
        splits=[(np.array(f['train_positions']),np.array(f['validation_positions'])) for f in m['inner']]
        inner_masks=[members if hallmark else masks[f['selector']] for f in m['inner']]
        best,rows,oof=tune(data,tr,splits,inner_masks,progress=lambda j,n:status('RF_TUNING',model=name,inner_fold=j,total_inner_folds=n,completed_models=number-1))
        pack=data.pack_selected(tr,te,members if hallmark else masks[m['outer_selector']])
        ridge=joblib.load(SOURCE/'models'/name/'classifier.joblib')
        assert ridge['training_PATNO']==meta.PATNO.iloc[tr].tolist() and ridge['test_PATNO']==meta.PATNO.iloc[te].tolist()
        np.testing.assert_array_equal(pack['state']['indices'],ridge['rna']['indices'])
        for key in ['means','scales']:np.testing.assert_allclose(pack['state'][key],ridge['rna'][key],atol=1e-12,rtol=0)
        prior=ridge_pred.loc[ridge_pred.model.eq(name)].set_index('PATNO').loc[meta.PATNO.iloc[te]]
        np.testing.assert_allclose(predict_artifact(ridge,raw[te]),prior.probability_PD,atol=1e-10,rtol=0)
        model=fit(pack['train'],y[tr],best);p=model.predict_proba(pack['test'])[:,1]
        state=artifact(model,pack,best,genes,[])
        state.update(algorithm='random_forest',model_name=name,family=m['family'],training_n=len(tr),training_PATNO=meta.PATNO.iloc[tr].tolist(),test_PATNO=meta.PATNO.iloc[te].tolist())
        if hallmark:state['predictors']='Fixed 4376 Hallmark member universe; training-eligible RNA genes'
        dest=ROOT/'models'/name;dest.mkdir(exist_ok=True);joblib.dump(state,dest/'classifier.joblib')
        np.testing.assert_allclose(predict_artifact(joblib.load(dest/'classifier.joblib'),raw[te]),p,atol=1e-12,rtol=0)
        importance=getattr(model,'feature_importances_',np.array([],float))
        pd.DataFrame(dict(Geneid=state['selected_gene_ids'],impurity_importance=importance,mean=pack['state']['means'],scale=pack['state']['scales'])).to_csv(dest/'feature_parameters.tsv',sep='\t',index=False)
        pd.DataFrame({'PATNO':meta.PATNO.iloc[tr].to_numpy(),'label':y[tr],**{'candidate_'+str(c):v for c,v in oof.items()}}).to_csv(dest/'inner_oof_predictions.tsv',sep='\t',index=False)
        fit_info=dict(model=name,family=m['family'],**best,training_n=len(tr),test_n=len(te),genes=pack['train'].shape[1],actual_min_samples_leaf=max(2,int(np.ceil(best['leaf_fraction']*len(tr)))),intercept_only=pack['train'].shape[1]==0)
        save(str(Path('models')/name/'fit_summary.json'),fit_info);chosen_rows.append(fit_info)
        scores.extend(dict(model=name,**r) for r in rows)
        for j,i in enumerate(te):predictions.append(dict(PATNO=meta.PATNO.iloc[i],global_index=int(i),stratum=meta.stratum.iloc[i],family=m['family'],model=name,algorithm='RF',label=int(y[i]),probability_PD=float(p[j]),threshold=best['threshold']))
        for mode,t in [('fixed_0_5',.5),('inner_selected',best['threshold'])]:metrics.append(dict(model=name,family=m['family'],threshold_mode=mode,**evaluate(y[te],p,t)))
        roundtrips.append(dict(model=name,genes_and_preprocessing_match='PASS',ridge_predictions_match='PASS',RF_artifact_predictions_match='PASS'))
        table('holdout_predictions.tsv',predictions);table('selected_parameters.tsv',chosen_rows);table('inner_tuning_scores.tsv',scores);table('model_metrics.tsv',metrics);table('model_validation.tsv',roundtrips)
    status('REPORTING')
    from reporting import report
    report(ROOT,meta,pd.DataFrame(predictions),ridge_pred,device)
    for p,digest in hashes.items():assert sha(p)==digest,p
    save('final_audit.json',dict(status='PASS',source_hashes_unchanged=len(hashes),DE_masks=123,models=13,matched_gene_sets=13,ridge_prediction_roundtrips=13,RF_prediction_roundtrips=13))
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['tmp','cache','matplotlib_cache']) and p.name not in ['status.json','output_sha256.json','.run.lock']]
    save('output_sha256.json',{str(p):sha(p) for p in files})
    status('COMPLETE',models=13,holdout_participants=105,elapsed_seconds=round(time.time()-start))

if __name__=='__main__':
    lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:status('FAILED',error=repr(e));traceback.print_exc();raise
