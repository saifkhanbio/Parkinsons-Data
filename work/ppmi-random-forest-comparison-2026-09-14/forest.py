"""Fixed RF grid with nested ANOVA gene selection and ridge controls."""
from pathlib import Path
import sys,io
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score

HERE=Path(__file__).resolve().parent
SELECTION=HERE.parent/'ppmi-mapped-gene-selection-2026-09-12'
STEP2=HERE.parent/'ppmi-ridge-demographic-models-2026-09-14'
STEP3=HERE.parent/'ppmi-ridge-class-weighting-2026-09-14'
sys.path.insert(0,str(SELECTION))
from gene_selection import (MAPPED,BASE,PRIOR,HALLMARK,MappedData,choose_device,make_splits,SEED,
    fit_model,select_threshold,evaluate,predict_artifact,rank,select)
ARMS=['original_ridge','selected_ridge','all_genes_rf','selected_rf']
KS=['100','500','all'];CS=[.001,.01,.1,1.]

def grid(ks=KS,cs=CS):
    candidates=[]
    for k in ks:
        for c in cs:candidates.append(dict(algorithm='ridge',k=k,C=c))
        for leaf in [25,10]:
            for mf in (['sqrt'] if k=='100' else ['sqrt',.1]):
                candidates.append(dict(algorithm='rf',k=k,min_samples_leaf=leaf,max_features=mf))
    return [dict(candidate=i,**r) for i,r in enumerate(candidates)]

GRID=grid()

def fit(x,y,config,trees=300):
    if config['algorithm']=='ridge':return fit_model(x,y,config['C'],'ridge',0.)
    model=RandomForestClassifier(n_estimators=trees,criterion='gini',bootstrap=True,max_samples=.8,max_depth=8,
        min_samples_leaf=config['min_samples_leaf'],max_features=config['max_features'],class_weight=None,
        random_state=SEED,n_jobs=4).fit(x,y)
    model.n_jobs=1
    assert model.classes_.tolist()==[0,1] and len(model.estimators_)==trees
    assert all(t.tree_.max_depth<=8 for t in model.estimators_)
    return model

def tie_key(r):
    return (-r['mean_inner_AUROC'],np.inf if r['k']=='all' else int(r['k']),
        r.get('C',0),-r.get('min_samples_leaf',0),r.get('max_features')!='sqrt')

def tune(data,train,splits,candidates=GRID,trees=300,progress=None):
    predictions={c['candidate']:np.full(len(train),np.nan) for c in candidates}
    scores={i:[] for i in predictions};aps={i:[] for i in predictions};seen=np.zeros(len(train),int);evidence={}
    for inner,(a,b) in enumerate(splits,1):
        assert not np.intersect1d(a,b).size and set(np.r_[a,b])==set(range(len(train)))
        seen[b]+=1;pack=data.pack(train[a],train[b]);order,f=rank(pack,data.labels[train[a]])
        evidence[f'{inner}_eligible_indices']=pack['state']['indices'];evidence[f'{inner}_F']=f
        evidence[f'{inner}_ranked_indices']=pack['state']['indices'][order]
        packs={k:select(pack,order,k)[0] for k in set(c['k'] for c in candidates)}
        for c in candidates:
            if progress:progress(inner,c)
            selected=packs[c['k']];model=fit(selected['train'],data.labels[train[a]],c,trees)
            p=model.predict_proba(selected['test'])[:,1];assert np.isfinite(p).all()
            i=c['candidate'];predictions[i][b]=p
            scores[i].append(float(roc_auc_score(data.labels[train[b]],p)))
            aps[i].append(float(average_precision_score(data.labels[train[b]],p)))
    assert (seen==1).all() and all(np.isfinite(p).all() for p in predictions.values())
    rows=[dict(**c,mean_inner_AUROC=float(np.mean(scores[c['candidate']])),mean_inner_AP=float(np.mean(aps[c['candidate']])),
          inner_AUROCs=';'.join(map(str,scores[c['candidate']])),inner_APs=';'.join(map(str,aps[c['candidate']]))) for c in candidates]
    best={}
    for arm in ARMS:
        algorithm='ridge' if 'ridge' in arm else 'rf'
        available=[r for r in rows if r['algorithm']==algorithm and (arm.startswith('selected') or r['k']=='all')]
        chosen=min(available,key=tie_key).copy()
        t,ba=select_threshold(data.labels[train],predictions[chosen['candidate']])
        chosen.update(arm=arm,threshold=t,inner_threshold_balanced_accuracy=ba);best[arm]=chosen
    return best,rows,predictions,evidence

def artifact(model,pack,chosen,genes):
    return dict(classifier=model,hyperparameters=chosen,threshold=chosen['threshold'],feature_universe=np.asarray(genes),
        rna=pack['state'],selected_gene_ids=np.asarray(genes)[pack['state']['indices']],positive_class='PD',
        selection='training-only ANOVA F' if chosen['k']!='all' else 'all eligible Hallmark candidates',
        transform='training-standardized sample-local log2(1+CPM)')

def preflight(device):
    import torch
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(90,150)).astype(np.int32);y=np.tile([0,1,1],30);raw[y==1,:5]+=25
    genes=np.arange(150).astype(str);indices=np.arange(120);tr=np.arange(72);te=np.arange(72,90)
    data=MappedData(raw,genes,y,indices,device);pack=data.pack(tr,te);order,f=rank(pack,y[tr])
    assert set(order[:5])==set(range(5))
    x0,x1=pack['train'][y[tr]==0],pack['train'][y[tr]==1]
    pooled=((len(x0)-1)*x0.var(0,ddof=1)+(len(x1)-1)*x1.var(0,ddof=1))/(len(tr)-2)
    np.testing.assert_allclose(f,(x1.mean(0)-x0.mean(0))**2/(pooled*(1/len(x0)+1/len(x1))),rtol=1e-10,atol=1e-10)
    duplicate=dict(train=np.c_[pack['train'][:,0],pack['train'][:,0]],state={'indices':np.array([8,2])})
    assert rank(duplicate,y[tr])[0].tolist()==[1,0]
    np.testing.assert_allclose(pack['train'],MappedData(raw,genes,y,indices,torch.device('cpu')).pack(tr,te)['train'],atol=1e-9,rtol=0)
    splits=make_splits(y[tr],tr,'participant_stratified',3,SEED)
    small=[dict(candidate=i,algorithm=a,k=k,**({'C':.01} if a=='ridge' else {'min_samples_leaf':10,'max_features':'sqrt'})) for i,(a,k) in enumerate([('ridge','5'),('ridge','all'),('rf','5'),('rf','all')])]
    best,rows,_,evidence=tune(data,tr,splits,small,trees=30)
    altered=raw.copy();altered[te]+=10000;other_y=y.copy();other_y[te]=1-other_y[te]
    other=MappedData(altered,genes,other_y,indices,device)
    other_best,other_rows,_,other_evidence=tune(other,tr,splits,small,trees=30)
    assert best==other_best and rows==other_rows
    for k,v in evidence.items():np.testing.assert_array_equal(v,other_evidence[k])
    for arm,chosen in best.items():
        selected,_=select(pack,order,chosen['k']);model=fit(selected['train'],y[tr],chosen,trees=30)
        buf=io.BytesIO();joblib.dump(artifact(model,selected,chosen,genes),buf);buf.seek(0)
        np.testing.assert_allclose(predict_artifact(joblib.load(buf),raw[te]),model.predict_proba(selected['test'])[:,1],atol=1e-9,rtol=0)
    centers=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    x=np.repeat(centers,100,axis=0)+rng.normal(0,.15,(400,2));yy=np.repeat([0,1,1,0],100)
    tr=np.concatenate([np.arange(i*100,i*100+75) for i in range(4)]);te=np.setdiff1d(np.arange(400),tr)
    model=fit(x[tr],yy[tr],dict(algorithm='rf',min_samples_leaf=10,max_features='sqrt'),trees=100)
    auc=roc_auc_score(yy[te],model.predict_proba(x[te])[:,1]);assert auc>.9
    return dict(heldout_ranking_tuning_threshold='PASS',ANOVA_t_squared='PASS',ranking_ties='PASS',
         CPU_device_preprocessing='PASS',serialized_inference='PASS',nonlinear_synthetic_AUROC=auc)
