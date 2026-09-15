"""RBF-SVM using the preserved, unsupervised mapped-gene preprocessing."""
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

WORK=Path(__file__).resolve().parent.parent
MAPPED=WORK/'ppmi-mapped-gene-classifier-2026-09-12'
SVM=WORK/'ppmi-svm-learning-curves-2026-09-12'
sys.path.insert(0,str(MAPPED)); sys.path.insert(0,str(SVM))
from mapped_genes import BASE,PRIOR,HALLMARK,MappedData,choose_device,make_splits,SEED,validate as validate_mapped
from svm_analysis import fit_model as fit_estimator,SVM_CS,GAMMAS


def fit(x,y,c,gamma_multiplier):
    return fit_estimator(x,y,'rbf_svm',c,gamma_multiplier)


def tune(data,train,splits,cs=None,gammas=None):
    cs=SVM_CS if cs is None else cs; gammas=GAMMAS if gammas is None else gammas
    scores={(c,g):[] for c in cs for g in gammas}; counts=[]; seen=np.zeros(len(train),dtype=int)
    for a,b in splits:
        assert not np.intersect1d(a,b).size
        seen[b]+=1; pack=data.pack(train[a],train[b]); counts.append(pack['train'].shape[1])
        for c,g in scores:
            model=fit(pack['train'],data.labels[train[a]],c,g)
            values=model.decision_function(pack['test']); assert np.isfinite(values).all()
            scores[(c,g)].append(float(roc_auc_score(data.labels[train[b]],values)))
    assert (seen==1).all()
    rows=[dict(C=c,gamma_multiplier=g,mean_inner_AUROC=float(np.mean(v)),inner_AUROCs=';'.join(map(str,v)),
               inner_eligible_counts=';'.join(map(str,counts))) for (c,g),v in scores.items()]
    best=min(rows,key=lambda r:(-r['mean_inner_AUROC'],r['C'],r['gamma_multiplier'])).copy()
    return best,rows


def artifact(model,pack,chosen,genes):
    return dict(classifier=model,rna=pack['state'],feature_universe=np.asarray(genes),
                selected_gene_ids=np.asarray(genes)[pack['state']['indices']],hyperparameters=chosen,
                positive_class='PD',score_type='uncalibrated_decision_margin',threshold=0.,
                threshold_description='Native SVM boundary; not clinically validated')


def predict_artifact(state,raw):
    raw=np.asarray(raw)
    assert raw.ndim==2 and raw.shape[1]==len(state['feature_universe'])
    assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
    library=raw.sum(1,dtype=np.float64); assert (library>0).all()
    s=state['rna']; x=np.log2(1+raw[:,s['indices']]/library[:,None]*1e6)
    result=state['classifier'].decision_function((x-s['means'])/s['scales'])
    assert np.isfinite(result).all()
    return result


def validate(device):
    checks=validate_mapped(device)
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(80,250)).astype(np.int32); y=np.tile([0,1],40); raw[y==1,:5]+=12
    genes=np.arange(250).astype(str); indices=np.arange(180); tr,te=np.arange(60),np.arange(60,80)
    data=MappedData(raw,genes,y,indices,device); splits=make_splits(y[tr],np.arange(60),'participant_stratified',3,SEED)
    best,_=tune(data,tr,splits,cs=[.1,1.],gammas=[.1,1.])
    modified_raw,modified_y=raw.copy(),y.copy(); modified_raw[te]+=10000; modified_y[te]=1-modified_y[te]
    other=MappedData(modified_raw,genes,modified_y,indices,device)
    changed,_=tune(other,tr,splits,cs=[.1,1.],gammas=[.1,1.]); assert best==changed
    pack=data.pack(tr,te); model=fit(pack['train'],y[tr],best['C'],best['gamma_multiplier'])
    state=artifact(model,pack,best,genes)
    np.testing.assert_allclose(predict_artifact(state,raw[te]),model.decision_function(pack['test']),atol=1e-9)
    centers=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    x=np.repeat(centers,40,axis=0)+rng.normal(0,.15,(160,2)); labels=np.repeat([0,1,1,0],40)
    train=np.concatenate([np.arange(j*40,j*40+30) for j in range(4)]); test=np.setdiff1d(np.arange(160),train)
    nonlinear=fit(x[train],labels[train],10.,1.)
    auc=roc_auc_score(labels[test],nonlinear.decision_function(x[test])); assert auc>.95
    checks.update(mapped_SVM_holdout_tuning='PASS',SVM_inference='PASS',synthetic_nonlinear_AUROC=float(auc))
    return checks
