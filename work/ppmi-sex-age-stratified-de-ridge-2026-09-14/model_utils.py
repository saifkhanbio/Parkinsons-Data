"""Benchmark-equivalent ridge with a training-selected gene-level FDR mask."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
BENCH=WORK/'ppmi-mapped-gene-classifier-2026-09-12'
BASE=WORK/'ppmi-classifier-2026-09-12'
PRIOR=WORK/'ppmi-classifier-refinement-retry-2026-09-12'
HALLMARK=WORK/'ppmi-hallmark-classifier-2026-09-12'
DE=WORK/'ppmi-deseq2-2026-09-12'
sys.path.insert(0,str(BENCH))
from mapped_genes import MappedData,fit_model,select_threshold,evaluate,make_splits,choose_device,SEED
CS=[.001,.01,.1,1.]

class ConstantModel:
    def __init__(self,y):
        self.prevalence_=float(np.mean(y))
        self.coef_=np.zeros((1,0));self.n_iter_=np.array([0]);self.classes_=np.array([0,1])
    def predict_proba(self,x):
        p=np.repeat(self.prevalence_,len(x))
        return np.column_stack([1-p,p])

def fit(x,y,c):
    return ConstantModel(y) if x.shape[1]==0 else fit_model(x,y,c,'ridge',0.)

class Data(MappedData):
    def pack_selected(self,train,test,allowed=None):
        pack=self.pack(train,test)
        if allowed is None: return pack
        keep=np.isin(pack['state']['indices'],np.asarray(allowed,dtype=int))
        return {'train':pack['train'][:,keep],'test':pack['test'][:,keep],
                'state':{k:v[keep] for k,v in pack['state'].items()}}

def tune(data,train,splits,masks):
    pred={c:np.full(len(train),np.nan) for c in CS}
    scores={c:[] for c in CS};seen=np.zeros(len(train),int);counts=[]
    assert len(splits)==len(masks)
    for (a,b),mask in zip(splits,masks):
        seen[b]+=1
        pack=data.pack_selected(train[a],train[b],mask);counts.append(pack['train'].shape[1])
        for c in CS:
            model=fit(pack['train'],data.labels[train[a]],c)
            p=model.predict_proba(pack['test'])[:,1];pred[c][b]=p
            scores[c].append(float(roc_auc_score(data.labels[train[b]],p)))
    assert (seen==1).all() and all(np.isfinite(x).all() for x in pred.values())
    rows=[dict(C=c,mean_inner_AUROC=float(np.mean(scores[c])),
               inner_AUROCs=';'.join(map(str,scores[c])),inner_eligible_counts=';'.join(map(str,counts))) for c in CS]
    best=min(rows,key=lambda r:(-r['mean_inner_AUROC'],r['C'])).copy()
    threshold,ba=select_threshold(data.labels[train],pred[best['C']])
    best.update(threshold=threshold,inner_threshold_balanced_accuracy=ba)
    return best,rows,pred

def artifact(model,pack,chosen,genes,pathways):
    return dict(classifier=model,hyperparameters=chosen,threshold=chosen['threshold'],
                feature_universe=np.asarray(genes),rna=pack['state'],
                selected_gene_ids=np.asarray(genes)[pack['state']['indices']],
                selected_pathways=pathways,positive_class='PD',
                transform='training-standardized sample-local log2(1+CPM)',
                predictors='training-selected DESeq2 gene-level FDR <0.05 and abs(log2FC)>0.5 RNA genes only',training_n=None)

def predict_artifact(state,raw):
    raw=np.asarray(raw)
    assert raw.ndim==2 and raw.shape[1]==len(state['feature_universe'])
    assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
    libs=raw.sum(1,dtype=np.float64);assert (libs>0).all()
    s=state['rna']
    x=(np.log2(1+raw[:,s['indices']]/libs[:,None]*1e6)-s['means'])/s['scales']
    return state['classifier'].predict_proba(x)[:,1]

def validate(device):
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(80,250)).astype(np.int32);y=np.tile([0,1],40)
    raw[y==1,5:10]+=12;raw[:,:2]=0
    genes=np.arange(250).astype(str);tr=np.arange(60);te=np.arange(60,80)
    data=Data(raw,genes,y,np.arange(250),device)
    splits=make_splits(y[tr],np.arange(60),'participant_stratified',3,SEED)
    masks=[np.arange(20),np.arange(30),np.array([],int)]
    best,rows,_=tune(data,tr,splits,masks)
    other_raw=raw.copy();other_raw[te]+=10000;other_y=y.copy();other_y[te]=1-other_y[te]
    other=Data(other_raw,genes,other_y,np.arange(250),device)
    best2,rows2,_=tune(other,tr,splits,masks)
    assert best==best2 and rows==rows2
    for mask in [np.array([5, 220, 249]),np.array([],int)]:
        a=data.pack_selected(tr,te,mask);b=other.pack_selected(tr,te,mask)
        np.testing.assert_array_equal(a['train'],b['train'])
        model=fit(a['train'],y[tr],.01)
        state=artifact(model,a,best,genes,[])
        np.testing.assert_allclose(predict_artifact(state,raw[te]),model.predict_proba(a['test'])[:,1],atol=1e-10)
        if not len(mask): assert np.all(predict_artifact(state,raw[te])==y[tr].mean())
    return dict(CPU_GPU_transform='PASS',heldout_perturbation='PASS',empty_selection='PASS',inference='PASS')
