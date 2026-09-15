"""Small prespecified forest grid on training-selected DE masks."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from model_utils import ConstantModel,Data,select_threshold,artifact,predict_artifact

GRID=[dict(candidate=i,leaf_fraction=leaf,max_features=mf) for i,(leaf,mf) in enumerate([(a,b) for a in [.05,.15] for b in ['sqrt',.3]])]

def fit(x,y,config,trees=300):
    if not x.shape[1]:return ConstantModel(y)
    model=RandomForestClassifier(n_estimators=trees,criterion='gini',bootstrap=True,max_samples=.8,max_depth=8,
        min_samples_leaf=max(2,int(np.ceil(config['leaf_fraction']*len(y)))),max_features=config['max_features'],
        class_weight=None,random_state=20260912,n_jobs=12)
    model.fit(x,y);model.n_jobs=1
    return model

def tune(data,train,splits,masks,trees=300,progress=None):
    pred={c['candidate']:np.full(len(train),np.nan) for c in GRID}
    scores={c['candidate']:[] for c in GRID};counts=[];seen=np.zeros(len(train),int)
    for j,((a,b),mask) in enumerate(zip(splits,masks),1):
        seen[b]+=1;pack=data.pack_selected(train[a],train[b],mask);counts.append(pack['train'].shape[1])
        for c in GRID:
            model=fit(pack['train'],data.labels[train[a]],c,trees)
            p=model.predict_proba(pack['test'])[:,1];pred[c['candidate']][b]=p
            scores[c['candidate']].append(float(roc_auc_score(data.labels[train[b]],p)))
        if progress:progress(j,len(splits))
    assert (seen==1).all() and all(np.isfinite(p).all() for p in pred.values())
    rows=[dict(**c,mean_inner_AUROC=float(np.mean(scores[c['candidate']])),inner_AUROCs=';'.join(map(str,scores[c['candidate']])),inner_eligible_counts=';'.join(map(str,counts))) for c in GRID]
    best=min(rows,key=lambda r:(-r['mean_inner_AUROC'],-r['leaf_fraction'],r['max_features']!='sqrt')).copy()
    threshold,ba=select_threshold(data.labels[train],pred[best['candidate']]);best.update(threshold=threshold,inner_threshold_balanced_accuracy=ba)
    return best,rows,pred

def validate(device):
    rng=np.random.default_rng(661);raw=rng.poisson(30,(80,100)).astype(np.int32);y=np.tile([0,1],40)
    raw[y==1,4:8]+=15;tr=np.arange(60);te=np.arange(60,80);genes=np.arange(100).astype(str)
    from sklearn.model_selection import StratifiedKFold
    splits=list(StratifiedKFold(3,shuffle=True,random_state=77).split(tr,y[tr]));masks=[np.arange(20),np.arange(30),np.array([],int)]
    data=Data(raw,genes,y,np.arange(100),device)
    a,ar,_=tune(data,tr,splits,masks,trees=20)
    modified=raw.copy();modified[te]+=9999;other_y=y.copy();other_y[te]=1-y[te]
    other=Data(modified,genes,other_y,np.arange(100),device)
    b,br,_=tune(other,tr,splits,masks,trees=20)
    assert a==b and ar==br
    for mask in [np.arange(20),np.array([],int)]:
        pack=data.pack_selected(tr,te,mask);model=fit(pack['train'],y[tr],a,trees=20)
        state=artifact(model,pack,a,genes,[])
        np.testing.assert_allclose(predict_artifact(state,raw[te]),model.predict_proba(pack['test'])[:,1],atol=1e-12)
        if not len(mask):assert np.all(predict_artifact(state,raw[te])==y[tr].mean())
    return dict(heldout_perturbation='PASS',empty_feature_folds='PASS',CPU_GPU_preprocessing='PASS',RF_inference='PASS')
