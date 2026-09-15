"""Training-only weighting selection for the original RNA-only ridge."""
from pathlib import Path
import sys,warnings,io
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score

HERE=Path(__file__).resolve().parent
MAPPED=HERE.parent/'ppmi-mapped-gene-classifier-2026-09-12'
STEP2=HERE.parent/'ppmi-ridge-demographic-models-2026-09-14'
sys.path.insert(0,str(MAPPED))
from mapped_genes import BASE,PRIOR,HALLMARK,MappedData,choose_device,make_splits,SEED,select_threshold,evaluate,predict_artifact

CS=[.001,.01,.1,1.]
ARMS=['unweighted','balanced','selected_weight']

def class_weights(y,weight):
    counts=np.bincount(y,minlength=2)
    assert (counts>0).all()
    return np.ones(2) if weight=='unweighted' else len(y)/(2*counts)

def fit(x,y,c,weight,sample_weight=None):
    assert weight in ['unweighted','balanced']
    for limit in [2000,10000]:
        model=LogisticRegression(C=c,penalty='l2',solver='lbfgs',max_iter=limit,tol=1e-7,
              class_weight=None if weight=='unweighted' else 'balanced',random_state=SEED)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(x,y,sample_weight=sample_weight)
        if not any(issubclass(m.category,ConvergenceWarning) for m in messages):
            assert model.classes_.tolist()==[0,1] and np.isfinite(model.coef_).all()
            return model
    raise RuntimeError('Ridge failed to converge after retry')

def tune(data,train,splits,cs=CS):
    keys=[(w,c) for w in ['unweighted','balanced'] for c in cs]
    predictions={k:np.full(len(train),np.nan) for k in keys}
    scores={k:[] for k in keys};aps={k:[] for k in keys};iterations={k:[] for k in keys}
    seen=np.zeros(len(train),int);weights=[];counts=[]
    for i,(a,b) in enumerate(splits,1):
        assert not np.intersect1d(a,b).size
        assert set(np.r_[a,b])==set(range(len(train)))
        seen[b]+=1;pack=data.pack(train[a],train[b]);y=data.labels[train[a]]
        counts.append(pack['train'].shape[1])
        for w,c in keys:
            model=fit(pack['train'],y,c,w)
            p=model.predict_proba(pack['test'])[:,1];assert np.isfinite(p).all()
            predictions[w,c][b]=p
            scores[w,c].append(float(roc_auc_score(data.labels[train[b]],p)))
            aps[w,c].append(float(average_precision_score(data.labels[train[b]],p)))
            iterations[w,c].append(int(model.n_iter_[0]))
        ww=class_weights(y,'balanced')
        weights.append(dict(inner_fold=i,n_train=len(y),n_control=int((y==0).sum()),n_PD=int(y.sum()),control_weight=ww[0],PD_weight=ww[1]))
    assert (seen==1).all() and all(np.isfinite(p).all() for p in predictions.values())
    rows=[dict(weight=w,C=c,mean_inner_AUROC=float(np.mean(scores[w,c])),mean_inner_AP=float(np.mean(aps[w,c])),
          inner_AUROCs=';'.join(map(str,scores[w,c])),inner_APs=';'.join(map(str,aps[w,c])),
          inner_eligible_counts=';'.join(map(str,counts)),maximum_optimizer_iterations=max(iterations[w,c])) for w,c in keys]
    best={}
    for arm in ARMS:
        candidates=rows if arm=='selected_weight' else [r for r in rows if r['weight']==arm]
        chosen=min(candidates,key=lambda r:(-r['mean_inner_AUROC'],r['weight']!='unweighted',r['C'])).copy()
        t,ba=select_threshold(data.labels[train],predictions[chosen['weight'],chosen['C']])
        chosen.update(arm=arm,threshold=t,inner_threshold_balanced_accuracy=ba);best[arm]=chosen
    return best,rows,predictions,weights

def artifact(model,pack,chosen,genes,y):
    return dict(classifier=model,hyperparameters=chosen,threshold=chosen['threshold'],
       feature_universe=np.asarray(genes),rna=pack['state'],selected_gene_ids=np.asarray(genes)[pack['state']['indices']],
       positive_class='PD',transform='training-standardized sample-local log2(1+CPM)',
       predictors='mapped RNA genes only',training_n=len(y),training_class_weights=class_weights(y,chosen['weight']))

def preflight(device):
    import torch
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(90,250)).astype(np.int32);raw[:,:2]=0
    y=np.tile([0,1,1],30);raw[y==1,5:10]+=10
    genes=np.arange(250).astype(str);indices=np.arange(180);tr=np.arange(72);te=np.arange(72,90)
    splits=make_splits(y[tr],tr,'participant_stratified',3,SEED)
    data=MappedData(raw,genes,y,indices,device);pack=data.pack(tr,te)
    best,rows,_,weights=tune(data,tr,splits,cs=[.01,.1])
    altered=raw.copy();altered[te]+=10000;other_y=y.copy();other_y[te]=1-other_y[te]
    other=MappedData(altered,genes,other_y,indices,device)
    other_best,other_rows,_,other_weights=tune(other,tr,splits,cs=[.01,.1])
    assert best==other_best and rows==other_rows and weights==other_weights
    np.testing.assert_array_equal(pack['train'],other.pack(tr,te)['train'])
    np.testing.assert_array_equal(pack['train'],MappedData(raw,genes,1-y,indices,device).pack(tr,te)['train'])
    cpu=MappedData(raw,genes,y,indices,torch.device('cpu')).pack(tr,te)
    np.testing.assert_allclose(pack['train'],cpu['train'],atol=1e-9,rtol=1e-9)
    for c in [.01,.1]:
        balanced=fit(pack['train'],y[tr],c,'balanced')
        explicit=fit(pack['train'],y[tr],c,'unweighted',sample_weight=class_weights(y[tr],'balanced')[y[tr]])
        np.testing.assert_allclose(balanced.predict_proba(pack['test']),explicit.predict_proba(pack['test']),atol=1e-10,rtol=0)
    for arm in ARMS:
        chosen=best[arm];model=fit(pack['train'],y[tr],chosen['C'],chosen['weight'])
        state=artifact(model,pack,chosen,genes,y[tr]);buf=io.BytesIO();joblib.dump(state,buf);buf.seek(0)
        np.testing.assert_allclose(predict_artifact(joblib.load(buf),raw[te]),model.predict_proba(pack['test'])[:,1],atol=1e-9,rtol=0)
    return dict(heldout_perturbation='PASS',training_only_preprocessing='PASS',CPU_device_agreement='PASS',
       balanced_equals_explicit_training_sample_weights='PASS',serialization_inference='PASS')
