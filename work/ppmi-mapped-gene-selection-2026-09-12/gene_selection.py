"""Nested ANOVA selection with an unchanged all-genes ridge control."""
from pathlib import Path
import sys
import tempfile
import joblib
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score

MAPPED=Path(__file__).resolve().parent.parent/'ppmi-mapped-gene-classifier-2026-09-12'
sys.path.insert(0,str(MAPPED))
from mapped_genes import (BASE,PRIOR,HALLMARK,MappedData,choose_device,make_splits,SEED,
    fit_model,select_threshold,evaluate,artifact as base_artifact,predict_artifact,validate as validate_mapping)

CS=[.001,.01,.1,1.]
KS=['100','500','1000','2000','all']


def rank(pack,y):
    scores=f_classif(pack['train'],y)[0]
    assert not np.isnan(scores).any() and (scores>=0).all()
    order=np.lexsort((pack['state']['indices'],-scores))
    return order,scores


def select(pack,order,k):
    if k=='all':
        return pack,np.arange(pack['train'].shape[1])
    count=int(k)
    assert count<=pack['train'].shape[1]
    local=np.sort(order[:count])
    selected=dict(train=pack['train'][:,local],test=pack['test'][:,local],
                  state={name:value[local] for name,value in pack['state'].items()})
    return selected,local


def tune(data,train,splits,ks=None,cs=None):
    ks=KS if ks is None else ks; cs=CS if cs is None else cs
    assert 'all' in ks
    keys=[(k,c) for k in ks for c in cs]
    scores={key:[] for key in keys}
    predictions={key:np.full(len(train),np.nan) for key in keys}
    seen=np.zeros(len(train),dtype=int)
    for a,b in splits:
        assert not np.intersect1d(a,b).size
        seen[b]+=1
        pack=data.pack(train[a],train[b])
        order,_=rank(pack,data.labels[train[a]])
        for k in ks:
            selected,_=select(pack,order,k)
            for c in cs:
                model=fit_model(selected['train'],data.labels[train[a]],c,'ridge',0.)
                p=model.predict_proba(selected['test'])[:,1]
                assert np.isfinite(p).all()
                predictions[(k,c)][b]=p
                scores[(k,c)].append(float(roc_auc_score(data.labels[train[b]],p)))
    assert (seen==1).all()
    rows=[dict(k=k,C=c,mean_inner_AUROC=float(np.mean(v)),inner_AUROCs=';'.join(map(str,v))) for (k,c),v in scores.items()]
    best={}
    for variant,candidates in [('original_ridge',[r for r in rows if r['k']=='all']),('selected_ridge',rows)]:
        chosen=min(candidates,key=lambda r:(-r['mean_inner_AUROC'],np.inf if r['k']=='all' else int(r['k']),r['C'])).copy()
        threshold,ba=select_threshold(data.labels[train],predictions[(chosen['k'],chosen['C'])])
        chosen.update(variant=variant,penalty='ridge',l1_ratio=0.,threshold=threshold,inner_threshold_balanced_accuracy=ba)
        best[variant]=chosen
    return best,rows


def artifact(model,selected,chosen,genes):
    state=base_artifact(model,selected,chosen,genes)
    state['selection_method']='training-only ANOVA F ranking; original index breaks ties'
    return state


def validate(device):
    checks=validate_mapping(device)
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(80,150)).astype(np.int32)
    y=np.tile([0,1],40); raw[y==1,:5]+=30
    genes,indices=np.arange(150).astype(str),np.arange(120)
    tr,te=np.arange(60),np.arange(60,80)
    data=MappedData(raw,genes,y,indices,device)
    pack=data.pack(tr,te); order,f=rank(pack,y[tr])
    assert set(order[:5])==set(range(5))
    x0,x1=pack['train'][y[tr]==0],pack['train'][y[tr]==1]
    pooled=((len(x0)-1)*x0.var(0,ddof=1)+(len(x1)-1)*x1.var(0,ddof=1))/(len(tr)-2)
    t_squared=(x1.mean(0)-x0.mean(0))**2/(pooled*(1/len(x0)+1/len(x1)))
    np.testing.assert_allclose(f,t_squared,rtol=1e-10,atol=1e-10)
    duplicate=dict(train=np.c_[pack['train'][:,0],pack['train'][:,0]],state={'indices':np.array([8,2])})
    assert rank(duplicate,y[tr])[0].tolist()==[1,0]
    all_pack,local=select(pack,order,'all')
    assert all_pack is pack and np.array_equal(local,np.arange(120))
    splits=make_splits(y[tr],np.arange(60),'participant_stratified',3,SEED)
    best,rows=tune(data,tr,splits,ks=['5','10','all'],cs=[.001,.1])
    changed_raw,labels=raw.copy(),y.copy(); changed_raw[te]+=10000; labels[te]=1-labels[te]
    changed=MappedData(changed_raw,genes,labels,indices,device)
    other,other_rows=tune(changed,tr,splits,ks=['5','10','all'],cs=[.001,.1])
    assert best==other and rows==other_rows
    np.testing.assert_array_equal(order,rank(changed.pack(tr,te),y[tr])[0])
    assert best['selected_ridge']['mean_inner_AUROC']>=best['original_ridge']['mean_inner_AUROC']
    chosen=best['selected_ridge']; selected,_=select(pack,order,chosen['k'])
    model=fit_model(selected['train'],y[tr],chosen['C'],'ridge',0.)
    with tempfile.TemporaryDirectory(prefix='gene_selection_check_') as folder:
        path=Path(folder)/'model.joblib'; joblib.dump(artifact(model,selected,chosen,genes),path)
        np.testing.assert_allclose(predict_artifact(joblib.load(path),raw[te]),model.predict_proba(selected['test'])[:,1],atol=1e-9)
    checks.update(ANOVA_t_squared='PASS',ranking_ties_and_synthetic_signal='PASS',
        holdout_ranking_tuning_threshold='PASS',all_genes_identity='PASS',selection_serialized_inference='PASS')
    return checks
