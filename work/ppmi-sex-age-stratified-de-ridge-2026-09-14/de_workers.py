"""Audited training-only R exports and restartable serial DE workers."""
from pathlib import Path
import json, os, subprocess, time, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'ppmi-classifier-2026-09-12'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def selected(frame):
    finite=np.isfinite(frame[['stat','pvalue','padj','log2FoldChange']].to_numpy(float)).all(1)
    return frame.loc[finite & frame.beta_converged.eq(True) & frame.padj.lt(.05) & frame.log2FoldChange.abs().gt(.5)]

def validate_result(dest,entry,genes):
    complete=json.loads((dest/'complete.json').read_text())
    assert complete['status']=='COMPLETE' and complete['training_n']==len(entry['train_indices'])
    assert json.loads((dest/'input.json').read_text())==entry
    for name,digest in json.loads((dest/'input_sha256.json').read_text()).items():
        assert sha(Path(name))==digest, name
    tab=pd.read_csv(dest/'de_results.tsv.gz',sep='\t')
    actual=pd.read_csv(dest/'selected_genes.tsv',sep='\t')
    assert tab.Geneid.is_unique and set(tab.Geneid)<=set(genes)
    assert selected(tab).Geneid.tolist()==actual.Geneid.tolist()
    assert len(actual)==complete['selected']
    finite=tab.padj.notna()
    p=tab.loc[finite,'pvalue'].to_numpy();q=tab.loc[finite,'padj'].to_numpy()
    order=np.argsort(p);bh=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1)
    np.testing.assert_allclose(bh,q[order],atol=1e-12,rtol=1e-9)
    return complete

def run_selector(entry):
    dest=ROOT/'selectors'/entry['name'];dest.mkdir(parents=True,exist_ok=True)
    genes=np.load(BASE/'gene_ids.npy')
    if (dest/'complete.json').exists():return validate_result(dest,entry,genes)
    raw=np.load(BASE/'counts.npy',mmap_mode='r')
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    idx=entry['train_indices'];tr=raw[idx]
    assert len(set(idx))==len(idx) and tr.dtype.kind in 'iu' and (tr>=0).all()
    assert tr.max()<np.iinfo(np.int32).max
    plan=json.loads((ROOT/'plan.json').read_text())
    assert not set(idx)&set(plan['test_indices'])
    frame=meta.iloc[idx][['PATNO','group','label']].copy()
    assert frame.PATNO.tolist()==entry['training_PATNO'] and frame.PATNO.is_unique
    frame['library_total']=tr.sum(1,dtype=np.int64)
    frame.to_csv(dest/'metadata.tsv',sep='\t',index=False)
    tr.astype('<i4').tofile(dest/'counts.int32');save(dest/'input.json',entry)
    paths=[dest/x for x in ['metadata.tsv','counts.int32','input.json']]+[ROOT/'run_selector.R',ROOT/'gene_ids.txt']
    save(dest/'input_sha256.json',{str(p):sha(p) for p in paths})
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with (dest/'run.log').open('w') as log:
        subprocess.run(['Rscript',str(ROOT/'run_selector.R'),str(ROOT),str(dest)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    result=validate_result(dest,entry,genes)
    save(dest/'output_sha256.json',{str(dest/x):sha(dest/x) for x in ['de_results.tsv.gz','selected_genes.tsv','prefilter.tsv','size_factors.tsv','complete.json']})
    return result

if __name__=='__main__':
    e=json.loads((ROOT/'plan.json').read_text())['selectors'][0]
    print(json.dumps(run_selector(e),indent=2),flush=True)
