"""Individual RNA predictors restricted to the fixed Hallmark gene union."""
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE=Path(__file__).resolve().parent.parent/'ppmi-classifier-2026-09-12'
PRIOR=BASE.parent/'ppmi-classifier-refinement-retry-2026-09-12'
HALLMARK=BASE.parent/'ppmi-hallmark-classifier-2026-09-12'
sys.path.insert(0,str(PRIOR))
from refine import fit_model,select_threshold,evaluate
from classifier import choose_device,make_splits,SEED

CS=[.001,.01,.1,1.]
RATIOS=[.1,.5,.9]
PENALTIES=['ridge','elasticnet']


class MappedData:
    def __init__(self,raw,genes,labels,indices,device):
        self.genes=np.asarray(genes); self.labels=np.asarray(labels)
        self.indices=np.asarray(indices,dtype=int); self.device=device
        assert raw.shape==(len(labels),len(genes))
        self.libraries=raw.sum(1,dtype=np.int64)
        assert (self.libraries>0).all()
        self.raw=np.asarray(raw[:,self.indices])
        self.raw_tensor=torch.as_tensor(self.raw,device=device)
        self.values=torch.log2(1+self.raw_tensor.to(torch.float64)/torch.as_tensor(self.libraries,device=device)[:,None]*1e6)
        self.checked_cpu=False

    def pack(self,train,test):
        train,test=np.asarray(train,dtype=int),np.asarray(test,dtype=int)
        assert not np.intersect1d(train,test).size
        tr,te=torch.as_tensor(train,device=self.device),torch.as_tensor(test,device=self.device)
        x=self.values[tr]; mean,sd=x.mean(0),x.std(0,unbiased=False)
        eligible=((self.raw_tensor[tr]>=10).sum(0)>=int(np.ceil(.2*len(train)))) & (sd>1e-12)
        selected=torch.nonzero(eligible).flatten(); local=selected.cpu().numpy()
        assert len(local)>0
        means,scales=mean[selected],sd[selected]
        a=((x[:,selected]-means)/scales).cpu().numpy()
        b=((self.values[te][:,selected]-means)/scales).cpu().numpy()
        state=dict(indices=self.indices[local],means=means.cpu().numpy(),scales=scales.cpu().numpy())
        if not self.checked_cpu:
            cpu=np.log2(1+self.raw[train][:,local]/self.libraries[train,None]*1e6)
            np.testing.assert_allclose(a,(cpu-cpu.mean(0))/cpu.std(0),atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(state['means'],cpu.mean(0),atol=1e-10)
            np.testing.assert_allclose(state['scales'],cpu.std(0),atol=1e-10)
            self.checked_cpu=True
        return dict(train=a,test=b,state=state)


def tune(data,train,splits,cs=None,ratios=None):
    cs=CS if cs is None else cs; ratios=RATIOS if ratios is None else ratios
    keys=[(p,c,r) for p in PENALTIES for c in cs for r in ([0.] if p=='ridge' else ratios)]
    predictions={k:np.full(len(train),np.nan) for k in keys}; scores={k:[] for k in keys}
    seen=np.zeros(len(train),dtype=int); counts=[]
    for a,b in splits:
        assert not np.intersect1d(a,b).size
        seen[b]+=1; pack=data.pack(train[a],train[b]); counts.append(pack['train'].shape[1])
        for key in keys:
            penalty,c,ratio=key
            model=fit_model(pack['train'],data.labels[train[a]],c,penalty,ratio)
            p=model.predict_proba(pack['test'])[:,1]; predictions[key][b]=p
            scores[key].append(float(roc_auc_score(data.labels[train[b]],p)))
    assert (seen==1).all()
    rows=[dict(penalty=p,C=c,l1_ratio=r,mean_inner_AUROC=float(np.mean(values)),inner_AUROCs=';'.join(map(str,values)),
               inner_eligible_counts=';'.join(map(str,counts))) for (p,c,r),values in scores.items()]
    best={}
    for penalty in PENALTIES:
        chosen=min([r for r in rows if r['penalty']==penalty],key=lambda r:(-r['mean_inner_AUROC'],r['C'],-r['l1_ratio'])).copy()
        p=predictions[(penalty,chosen['C'],chosen['l1_ratio'])]; assert np.isfinite(p).all()
        threshold,ba=select_threshold(data.labels[train],p)
        chosen.update(threshold=threshold,inner_threshold_balanced_accuracy=ba); best[penalty]=chosen
    return best,rows


def artifact(model,pack,chosen,genes):
    return dict(classifier=model,penalty=chosen['penalty'],hyperparameters=chosen,threshold=chosen['threshold'],
                feature_universe=np.asarray(genes),rna=pack['state'],selected_gene_ids=np.asarray(genes)[pack['state']['indices']],
                positive_class='PD',transform='training-standardized sample-local log2(1+CPM)',predictors='mapped RNA genes only')


def predict_artifact(state,raw):
    raw=np.asarray(raw)
    assert raw.ndim==2 and raw.shape[1]==len(state['feature_universe'])
    assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
    library=raw.sum(1,dtype=np.float64); assert (library>0).all()
    s=state['rna']; values=np.log2(1+raw[:,s['indices']]/library[:,None]*1e6)
    return state['classifier'].predict_proba((values-s['means'])/s['scales'])[:,1]


def validate(device):
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(80,250)).astype(np.int32); raw[:,:2]=0
    y=np.tile([0,1],40); raw[y==1,5:10]+=12
    genes=np.arange(250).astype(str); indices=np.arange(180)
    tr,te=np.arange(60),np.arange(60,80)
    data=MappedData(raw,genes,y,indices,device); a=data.pack(tr,te)
    assert set(a['state']['indices'])==set(range(2,180))
    splits=make_splits(y[tr],np.arange(60),'participant_stratified',3,SEED)
    best,_=tune(data,tr,splits,cs=[.01,.1],ratios=[.5])
    altered_raw,altered_y=raw.copy(),y.copy(); altered_raw[te]+=10000; altered_y[te]=1-altered_y[te]
    other=MappedData(altered_raw,genes,altered_y,indices,device); b=other.pack(tr,te)
    other_best,_=tune(other,tr,splits,cs=[.01,.1],ratios=[.5]); assert best==other_best
    np.testing.assert_array_equal(a['train'],b['train'])
    reversed_data=MappedData(raw,genes,1-y,indices,device)
    np.testing.assert_array_equal(a['train'],reversed_data.pack(tr,te)['train'])
    for penalty,chosen in best.items():
        model=fit_model(a['train'],y[tr],chosen['C'],penalty,chosen['l1_ratio'])
        state=artifact(model,a,chosen,genes)
        np.testing.assert_allclose(predict_artifact(state,raw[te]),model.predict_proba(a['test'])[:,1],atol=1e-9)
    return dict(candidate_filter='PASS',CPU_GPU_agreement='PASS',heldout_perturbation_tuning_threshold='PASS',
                label_independent_preprocessing='PASS',artifact_inference='PASS')
