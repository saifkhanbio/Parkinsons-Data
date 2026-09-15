"""Fold-local SVM/ridge fitting and deterministic learning subsets."""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent/'ppmi-classifier-2026-09-12'
sys.path.insert(0,str(BASE))
from classifier import (FoldData, choose_device, make_splits, matrices, artifact,
                        clinical_matrix, fit_logistic, validate_leakage, CS, KS, SEED)

FAMILIES = ['blood_clinical','rna','combined']
ALGORITHMS = ['ridge','rbf_svm']
SVM_CS = [.1,1.,10.,100.]
GAMMAS = [.01,.1,1.,10.]
FRACTIONS = [.25,.5,.75,1.]


def score(model,x,algorithm):
    s = model.predict_proba(x)[:,1] if algorithm=='ridge' else model.decision_function(x)
    assert np.isfinite(s).all()
    return s


def fit_model(x,y,algorithm,c,gamma_multiplier=0.):
    if algorithm=='ridge':
        return fit_logistic(x,y,c)
    assert algorithm=='rbf_svm' and gamma_multiplier>0
    model = SVC(C=c,gamma=gamma_multiplier/x.shape[1],kernel='rbf',probability=False,
                class_weight=None,tol=1e-4,max_iter=-1,cache_size=256,random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y)
    assert model.fit_status_==0 and model.classes_.tolist()==[0,1]
    assert np.isfinite(model.dual_coef_).all()
    return model


def tune(data,train,splits,families=None,algorithms=None,ks=None,ridge_cs=None,svm_cs=None,gammas=None):
    families = FAMILIES if families is None else families
    algorithms = ALGORITHMS if algorithms is None else algorithms
    ks = KS if ks is None else ks
    ridge_cs = CS if ridge_cs is None else ridge_cs
    svm_cs = SVM_CS if svm_cs is None else svm_cs
    gammas = GAMMAS if gammas is None else gammas
    keys = [(f,a,k,c,g) for f in families for a in algorithms
            for k in (ks if f in ['rna','combined'] else [0])
            for c in (ridge_cs if a=='ridge' else svm_cs)
            for g in ([0.] if a=='ridge' else gammas)]
    scores = {key:[] for key in keys}
    seen = np.zeros(len(train),dtype=int)
    for a,b in splits:
        assert not np.intersect1d(a,b).size
        seen[b]+=1
        tr,va = train[a],train[b]
        pack = data.pack(tr,va)
        for key in keys:
            f,algorithm,k,c,g = key
            x,v = matrices(pack,f,k)
            model = fit_model(x,data.labels[tr],algorithm,c,g)
            scores[key].append(float(roc_auc_score(data.labels[va],score(model,v,algorithm))))
    assert (seen==1).all()
    rows = [dict(family=f,algorithm=a,k=k,C=c,gamma_multiplier=g,mean_inner_AUROC=float(np.mean(values)),
                 inner_AUROCs=';'.join(map(str,values))) for (f,a,k,c,g),values in scores.items()]
    best = {}
    for f in families:
        for a in algorithms:
            candidates = [r for r in rows if r['family']==f and r['algorithm']==a]
            best[(f,a)] = min(candidates,key=lambda r:(-r['mean_inner_AUROC'],r['k'],r['C'],r['gamma_multiplier'])).copy()
    return best,rows


def subset_plan(train,y,batches,protocol,seed,original_inner):
    rng = np.random.default_rng(seed+700001)
    class_orders = {c:rng.permutation(train[y[train]==c]) for c in [0,1]}
    group_order = rng.permutation(np.unique(batches[train]))
    previous,plans = set(),[]
    for j,fraction in enumerate(FRACTIONS):
        if fraction==1.:
            subset = train.copy()
            splits = [(np.array(s['train_positions']),np.array(s['validation_positions'])) for s in original_inner]
        elif protocol=='participant_stratified':
            subset = np.sort(np.concatenate([order[:int(np.ceil(fraction*len(order)))] for order in class_orders.values()]))
            splits = make_splits(y[subset],batches[subset],protocol,3,seed+1000*(j+1))
        else:
            for n_groups in range(3,len(group_order)+1):
                subset = np.sort(train[np.isin(batches[train],group_order[:n_groups])])
                if len(subset)<int(np.ceil(fraction*len(train))) or not previous.issubset(set(subset)):
                    continue
                try:
                    splits = make_splits(y[subset],batches[subset],protocol,3,seed+1000*(j+1))
                    assert all(min(np.bincount(y[subset[a]],minlength=2))>1 for a,b in splits)
                    break
                except (AssertionError,ValueError):
                    continue
            else:
                raise AssertionError('Cannot construct valid grouped learning subset')
        assert previous.issubset(set(subset)) and set(subset).issubset(set(train))
        for a,b in splits:
            assert set(np.r_[a,b])==set(range(len(subset))) and not np.intersect1d(a,b).size
            assert min(np.bincount(y[subset[a]],minlength=2))>1
            assert len(np.unique(y[subset[b]]))==2
            if protocol=='batch_grouped':
                assert not set(batches[subset[a]]) & set(batches[subset[b]])
        previous = set(subset)
        plans.append(dict(fraction=fraction,train_indices=subset.tolist(),n_train=len(subset),
                          n_train_groups=len(np.unique(batches[subset])),PD=int(y[subset].sum()),Control=int((y[subset]==0).sum()),
                          inner=[dict(train_positions=a.tolist(),validation_positions=b.tolist()) for a,b in splits]))
    assert plans[-1]['train_indices']==train.tolist()
    return plans


def predict_artifact(state,raw,metadata):
    raw = np.asarray(raw)
    assert raw.ndim==2 and raw.shape==(len(metadata),len(state['feature_universe']))
    assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
    libraries = raw.sum(1,dtype=np.float64)
    assert (libraries>0).all()
    pieces = []
    if 'clinical' in state:
        s = state['clinical']; x = clinical_matrix(metadata,state['clinical_family'],libraries)
        pieces.append((x-s['means'])/s['scales'])
    if 'rna' in state:
        s = state['rna']; x = np.log2(1+raw[:,s['indices']]/libraries[:,None]*1e6)
        pieces.append((x-s['means'])/s['scales'])
    x = np.column_stack(pieces)
    return score(state['classifier'],x,state['algorithm'])


def validate_svm(device):
    checks = validate_leakage(device)
    rng = np.random.default_rng(SEED)
    # A separate synthetic XOR example verifies the nonlinear learner's behavior.
    centers = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    x = np.repeat(centers,40,axis=0)+rng.normal(0,.15,(160,2))
    y = np.repeat([0,1,1,0],40)
    train = np.concatenate([np.arange(i*40,i*40+30) for i in range(4)])
    test = np.setdiff1d(np.arange(160),train)
    model = fit_model(x[train],y[train],'rbf_svm',10.,1.)
    auc = roc_auc_score(y[test],score(model,x[test],'rbf_svm'))
    assert auc>.95
    raw = rng.poisson(30,(80,150)).astype(np.int32)
    y = np.tile([0,1],40); raw[y==1,:5]+=10
    meta = pd.DataFrame(dict(age_collection_years=rng.normal(60,8,80),sex=np.where(y,'Male','Female'),
             wbc=rng.uniform(4,8,80),neutrophils_percent=rng.uniform(40,70,80),monocytes_percent=rng.uniform(3,8,80),
             eosinophils_percent=rng.uniform(1,3,80),basophils_percent=rng.uniform(.1,1,80),RIN=rng.uniform(6,9,80),
             intergenic_percent=rng.uniform(5,15,80),phase='PPMI-Phase1'))
    genes = np.arange(150).astype(str)
    tr,te = np.arange(60),np.arange(60,80)
    splits = make_splits(y[tr],np.arange(60),'participant_stratified',3,SEED)
    data = FoldData(raw,genes,meta,y,device)
    kwargs = dict(families=['combined'],ks=[100],ridge_cs=[.01,.1],svm_cs=[.1,1.],gammas=[.1,1.])
    best,_ = tune(data,tr,splits,**kwargs)
    altered_raw,altered_y,altered_meta = raw.copy(),y.copy(),meta.copy()
    altered_raw[te]+=10000; altered_y[te]=1-altered_y[te]; altered_meta.loc[te,'age_collection_years']+=1000
    altered = FoldData(altered_raw,genes,altered_meta,altered_y,device)
    other,_ = tune(altered,tr,splits,**kwargs)
    assert best==other
    pack = data.pack(tr,te)
    b = best[('combined','rbf_svm')]
    a,v = matrices(pack,'combined',b['k'])
    model = fit_model(a,y[tr],'rbf_svm',b['C'],b['gamma_multiplier'])
    state = artifact(model,pack,'combined',b['k'],genes); state['algorithm']='rbf_svm'
    np.testing.assert_allclose(predict_artifact(state,raw[te],meta.iloc[te]),score(model,v,'rbf_svm'),atol=1e-9)
    checks.update(synthetic_nonlinear_XOR_AUROC=float(auc),heldout_perturbation_tuning='PASS',svm_artifact_inference='PASS')
    return checks
