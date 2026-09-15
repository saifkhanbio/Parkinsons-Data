"""Training-only, equal-weight Hallmark expression features and logistic tuning."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

BASE=Path(__file__).resolve().parent.parent/'ppmi-classifier-2026-09-12'
PRIOR=BASE.parent/'ppmi-classifier-refinement-retry-2026-09-12'
sys.path.insert(0,str(PRIOR))
from refine import fit_model,select_threshold,evaluate
from classifier import clinical_matrix,choose_device,make_splits,SEED

FAMILIES=['blood_clinical','pathways','combined']
PENALTIES=['ridge','elasticnet']
CS=[.001,.01,.1,1.]
RATIOS=[.1,.5,.9]


class PathwayData:
    def __init__(self,raw,genes,meta,labels,indices,membership,names,device):
        self.labels=np.asarray(labels); self.genes=np.asarray(genes)
        self.meta=meta; self.indices=np.asarray(indices,dtype=int)
        self.names=np.asarray(names); self.device=device
        assert raw.shape==(len(labels),len(genes)) and len(names)==50
        libraries=raw.sum(1,dtype=np.int64)
        assert (libraries>0).all()
        self.raw=np.asarray(raw[:,self.indices])
        self.membership=np.asarray(membership,dtype=float)
        assert self.membership.shape==(len(indices),50) and (self.membership.sum(0)>=5).all()
        self.raw_tensor=torch.as_tensor(self.raw,device=device)
        self.rna=torch.log2(1+self.raw_tensor.to(torch.float64)/torch.as_tensor(libraries,device=device)[:,None]*1e6)
        self.membership_tensor=torch.as_tensor(self.membership,device=device)
        self.clinical=clinical_matrix(meta,'blood_clinical',libraries)
        self.coverage=[]; self.checked_cpu=False
        self.libraries=libraries

    def pack(self,train,test,tag=None):
        train,test=np.asarray(train,dtype=int),np.asarray(test,dtype=int)
        assert not np.intersect1d(train,test).size
        tr=torch.as_tensor(train,device=self.device); te=torch.as_tensor(test,device=self.device)
        a=self.rna[tr]; mean=a.mean(0); sd=a.std(0,unbiased=False)
        keep=((self.raw_tensor[tr]>=10).sum(0)>=int(np.ceil(.2*len(train)))) & (sd>1e-12)
        eligible=torch.nonzero(keep).flatten()
        membership=self.membership_tensor[eligible]
        sizes=membership.sum(0)
        assert (sizes>=5).all(),'A Hallmark set has fewer than five training-eligible genes'
        weights=membership/sizes
        z=(a[:,eligible]-mean[eligible])/sd[eligible]
        v=(self.rna[te][:,eligible]-mean[eligible])/sd[eligible]
        scores=z@weights; test_scores=v@weights
        pm,ps=scores.mean(0),scores.std(0,unbiased=False)
        constant=ps<1e-12; ps=torch.where(constant,torch.ones_like(ps),ps)
        tx=((scores-pm)/ps).cpu().numpy(); vx=((test_scores-pm)/ps).cpu().numpy()
        local=eligible.cpu().numpy()
        state=dict(indices=self.indices[local],gene_means=mean[eligible].cpu().numpy(),gene_scales=sd[eligible].cpu().numpy(),
                   weights=weights.cpu().numpy(),pathway_means=pm.cpu().numpy(),pathway_scales=ps.cpu().numpy(),names=self.names)
        if not self.checked_cpu:
            cpu=np.log2(1+self.raw[train][:,local]/self.libraries[train,None]*1e6)
            cm,cs=cpu.mean(0),cpu.std(0)
            standardized=(cpu-cm)/cs
            cpu_scores=np.column_stack([standardized[:,self.membership[local,j]>0].mean(1) for j in range(50)])
            cps=cpu_scores.std(0); cps[cps<1e-12]=1
            np.testing.assert_allclose(tx,(cpu_scores-cpu_scores.mean(0))/cps,atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(state['gene_means'],cm,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(state['gene_scales'],cs,atol=1e-10,rtol=1e-10)
            self.checked_cpu=True
        mu,scale=self.clinical[train].mean(0),self.clinical[train].std(0)
        scale[scale<1e-12]=1
        clinical=dict(train=(self.clinical[train]-mu)/scale,test=(self.clinical[test]-mu)/scale,means=mu,scales=scale)
        if tag is not None:
            self.coverage.extend(dict(partition=tag,pathway=str(name),eligible_genes=int(count),mapped_genes=int(self.membership[:,j].sum()),
                                      constant_score=bool(constant[j].item())) for j,(name,count) in enumerate(zip(self.names,sizes.cpu().numpy())))
        return dict(path_train=tx,path_test=vx,pathway_state=state,clinical=clinical)


def matrices(pack,family):
    if family=='blood_clinical':
        return pack['clinical']['train'],pack['clinical']['test']
    if family=='pathways':
        return pack['path_train'],pack['path_test']
    assert family=='combined'
    return np.column_stack([pack['clinical']['train'],pack['path_train']]),np.column_stack([pack['clinical']['test'],pack['path_test']])


def tune(data,train,splits,tag='tuning',cs=None,ratios=None):
    cs=CS if cs is None else cs; ratios=RATIOS if ratios is None else ratios
    keys=[(f,p,c,r) for f in FAMILIES for p in PENALTIES for c in cs for r in ([0.] if p=='ridge' else ratios)]
    predictions={key:np.full(len(train),np.nan) for key in keys}; scores={key:[] for key in keys}
    seen=np.zeros(len(train),dtype=int)
    for fold,(a,b) in enumerate(splits,1):
        seen[b]+=1
        pack=data.pack(train[a],train[b],f'{tag}_inner_{fold}')
        for key in keys:
            family,penalty,c,ratio=key
            x,v=matrices(pack,family)
            model=fit_model(x,data.labels[train[a]],c,penalty,ratio)
            p=model.predict_proba(v)[:,1]; predictions[key][b]=p
            scores[key].append(float(roc_auc_score(data.labels[train[b]],p)))
    assert (seen==1).all()
    rows=[dict(family=f,penalty=p,C=c,l1_ratio=r,mean_inner_AUROC=float(np.mean(s)),inner_AUROCs=';'.join(map(str,s))) for (f,p,c,r),s in scores.items()]
    best={}
    for f in FAMILIES:
        for penalty in PENALTIES:
            chosen=min([r for r in rows if r['family']==f and r['penalty']==penalty],key=lambda r:(-r['mean_inner_AUROC'],r['C'],-r['l1_ratio'])).copy()
            p=predictions[(f,penalty,chosen['C'],chosen['l1_ratio'])]
            assert np.isfinite(p).all()
            threshold,ba=select_threshold(data.labels[train],p)
            chosen.update(threshold=threshold,inner_threshold_balanced_accuracy=ba)
            best[(f,penalty)]=chosen
    return best,rows


def artifact(model,pack,family,penalty,chosen,genes):
    state=dict(classifier=model,family=family,penalty=penalty,threshold=chosen['threshold'],hyperparameters=chosen,
               feature_universe=np.asarray(genes),positive_class='PD',transform='mean training-standardized log2(1+CPM) Hallmark members')
    if family!='blood_clinical':
        state['pathway_state']=pack['pathway_state']
    if family!='pathways':
        state['clinical']={k:pack['clinical'][k] for k in ['means','scales']}
    return state


def predict_artifact(state,raw,meta):
    raw=np.asarray(raw)
    assert raw.shape==(len(meta),len(state['feature_universe'])) and np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
    library=raw.sum(1,dtype=np.float64); assert (library>0).all()
    parts=[]
    if 'clinical' in state:
        s=state['clinical']; x=clinical_matrix(meta,'blood_clinical',library)
        parts.append((x-s['means'])/s['scales'])
    if 'pathway_state' in state:
        s=state['pathway_state']
        x=np.log2(1+raw[:,s['indices']]/library[:,None]*1e6)
        values=((x-s['gene_means'])/s['gene_scales'])@s['weights']
        parts.append((values-s['pathway_means'])/s['pathway_scales'])
    return state['classifier'].predict_proba(np.column_stack(parts))[:,1]


def validate(device):
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(80,180)).astype(np.int32); raw[:,:2]=0
    genes=np.arange(180).astype(str); y=np.tile([0,1],40)
    meta=pd.DataFrame(dict(age_collection_years=rng.normal(60,8,80),sex=np.where(y,'Male','Female'),wbc=rng.uniform(4,8,80),
         neutrophils_percent=rng.uniform(40,70,80),monocytes_percent=rng.uniform(3,8,80),eosinophils_percent=rng.uniform(1,3,80),basophils_percent=rng.uniform(.1,1,80)))
    membership=np.zeros((180,50))
    for j in range(50): membership[rng.choice(180,20,replace=False),j]=1
    names=np.array([f'test_pathway_{j}' for j in range(50)])
    tr,te=np.arange(60),np.arange(60,80)
    data=PathwayData(raw,genes,meta,y,np.arange(180),membership,names,device)
    a=data.pack(tr,te)
    splits=make_splits(y[tr],np.arange(60),'participant_stratified',3,SEED)
    best,_=tune(data,tr,splits,cs=[.01,.1],ratios=[.5])
    other_raw,other_y,other_meta=raw.copy(),y.copy(),meta.copy()
    other_raw[te]+=10000; other_y[te]=1-other_y[te]; other_meta.loc[te,'age_collection_years']+=1000
    other=PathwayData(other_raw,genes,other_meta,other_y,np.arange(180),membership,names,device)
    b=other.pack(tr,te); other_best,_=tune(other,tr,splits,cs=[.01,.1],ratios=[.5])
    assert best==other_best
    np.testing.assert_array_equal(a['path_train'],b['path_train'])
    # Change every label: unsupervised pathway preprocessing must still be identical.
    reversed_data=PathwayData(raw,genes,meta,1-y,np.arange(180),membership,names,device)
    np.testing.assert_array_equal(a['path_train'],reversed_data.pack(tr,te)['path_train'])
    for (family,penalty),chosen in best.items():
        x,v=matrices(a,family)
        model=fit_model(x,y[tr],chosen['C'],penalty,chosen['l1_ratio'])
        state=artifact(model,a,family,penalty,chosen,genes)
        np.testing.assert_allclose(predict_artifact(state,raw[te],meta.iloc[te]),model.predict_proba(v)[:,1],atol=1e-9)
    return dict(pathway_CPU_GPU_agreement='PASS',heldout_perturbation_preprocessing_tuning_threshold='PASS',
                label_independent_pathway_preprocessing='PASS',all_six_artifact_inference_checks='PASS')
