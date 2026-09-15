"""Training-specific length-normalized, non-directional singscore features."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent;MAPPED=ROOT.parent/'ppmi-mapped-gene-classifier-2026-09-12'
sys.path.insert(0,str(MAPPED))
from mapped_genes import BASE,HALLMARK,choose_device,fit_model,select_threshold,evaluate
PATH_FAMILIES=['Hallmark_score','C2_CP_score','C5_BP_score']
FAMILIES=['rna','rna_blood']+PATH_FAMILIES
CS=[.001,.01,.1,1.]

class ConstantModel:
    def __init__(self,y):self.prevalence_=float(np.mean(y));self.coef_=np.zeros((1,0));self.n_iter_=np.array([0])
    def predict_proba(self,x):
        p=np.repeat(self.prevalence_,len(x));return np.column_stack([1-p,p])

def fit(x,y,c):return ConstantModel(y) if not x.shape[1] else fit_model(x,y,c,'ridge',0.)

def folded_ranks(x):
    values,order=torch.sort(x,dim=1,stable=True);n=x.shape[1]
    boundary=torch.cat([torch.ones((len(x),1),device=x.device,dtype=torch.bool),values[:,1:]!=values[:,:-1]],dim=1)
    positions=torch.arange(1,n+1,device=x.device,dtype=torch.float64).expand_as(values)
    ranks=torch.cummax(torch.where(boundary,positions,torch.zeros_like(positions)),dim=1).values
    med=torch.ceil((ranks[:,(n-1)//2]+ranks[:,n//2])/2)
    folded=torch.abs(ranks-med[:,None]);out=torch.empty_like(folded);out.scatter_(1,order,folded)
    return out

def score_tensor(rpk,membership,device):
    n=rpk.shape[1];m=np.asarray(membership.sum(1)).ravel();assert (m>0).all() and (m<n).all()
    mat=membership.tocoo();weights=mat.data/m[mat.row]
    index=torch.tensor(np.vstack([mat.row,mat.col]),device=device,dtype=torch.long)
    value=torch.tensor(weights,device=device,dtype=torch.float64)
    tensor=torch.sparse_coo_tensor(index,value,mat.shape,device=device,dtype=torch.float64).coalesce()
    avg=torch.sparse.mm(tensor,folded_ranks(rpk).T).T
    k=np.floor(m/2);B=np.ceil(n/2);lo=(k+1)/2;den=B-k
    return (avg-torch.tensor(lo,device=device))/(torch.tensor(den,device=device))

def cpu_scores(raw,length,membership):
    x=np.asarray(raw,dtype=float)/np.asarray(length)[None,:];r=rankdata(x,method='min',axis=1)
    z=abs(r-np.ceil(np.median(r,axis=1))[:,None]);m=np.asarray(membership.sum(1)).ravel();k=np.floor(m/2)
    return (np.asarray(membership@z.T).T/m[None,:]-(k+1)/2)/(np.ceil(x.shape[1]/2)-k)

class PathData:
    def __init__(self,raw,genes,labels,length,unambiguous,membership,reference,device):
        self.raw=raw;self.genes=genes;self.labels=labels;self.length=length;self.unambiguous=unambiguous
        self.membership=membership;self.reference=reference;self.device=device
        self.counts=torch.as_tensor(np.asarray(raw),device=device);self.rpk=self.counts.to(torch.float64)/torch.tensor(length,device=device)
        self.checked=False
    def pack(self,train,test):
        train=np.asarray(train,int);test=np.asarray(test,int);assert not set(train)&set(test)
        mask=((self.counts[torch.tensor(train,device=self.device)]>=10).sum(0)>=int(np.ceil(.2*len(train)))).cpu().numpy()&self.unambiguous
        background=np.flatnonzero(mask);assert len(background)>500
        member=self.membership[:,background];sizes=np.asarray(member.sum(1)).ravel();eligible=(sizes>=15)&(sizes<=500)&(sizes<len(background))
        selected=np.flatnonzero(eligible);local=member[selected];samples=np.r_[train,test]
        values=score_tensor(self.rpk[torch.tensor(samples,device=self.device)][:,torch.tensor(background,device=self.device)],local,self.device)
        if not self.checked:
            wanted=np.unique(np.r_[np.arange(min(20,len(selected))),np.arange(max(0,len(selected)-20),len(selected))])
            ref=cpu_scores(self.raw[samples[:3]][:,background],self.length[background],local[wanted])
            np.testing.assert_allclose(values[:3,wanted].cpu().numpy(),ref,atol=1e-11,rtol=1e-11);self.checked=True
        means=values[:len(train)].mean(0);sd=values[:len(train)].std(0,unbiased=False);variable=(sd>1e-12).cpu().numpy()
        result={};coverage=self.reference[['pathway_index','collection','pathway']].copy();coverage['eligible_members']=sizes;coverage['size_eligible']=eligible;coverage['nonconstant_score']=False
        coverage.loc[selected[variable],'nonconstant_score']=True
        for family in PATH_FAMILIES:
            collection=family.removesuffix('_score') if hasattr(family,'removesuffix') else family[:-6]
            keep=np.flatnonzero((self.reference.collection.iloc[selected].to_numpy()==collection)&variable)
            col=torch.tensor(keep,device=self.device);v=(values[:,col]-means[col])/sd[col]
            result[family]=dict(train=v[:len(train)].cpu().numpy(),test=v[len(train):].cpu().numpy(),
                state=dict(background_indices=background,pathway_indices=selected[keep],membership=local[keep].tocsr(),means=means[col].cpu().numpy(),scales=sd[col].cpu().numpy(),pathway_names=self.reference.pathway.iloc[selected[keep]].to_numpy(),lengths=self.length[background]))
        return result,coverage

def tune(data,train,splits):
    keys=[(f,c) for f in PATH_FAMILIES for c in CS];pred={k:np.full(len(train),np.nan) for k in keys};aucs={k:[] for k in keys};aps={k:[] for k in keys};counts={f:[] for f in PATH_FAMILIES};seen=np.zeros(len(train),int);eligibility=[]
    for j,(a,b) in enumerate(splits,1):
        seen[b]+=1;packs,coverage=data.pack(train[a],train[b]);coverage['inner_fold']=j;eligibility.append(coverage)
        for f in PATH_FAMILIES:
            pack=packs[f];counts[f].append(pack['train'].shape[1])
            for c in CS:
                model=fit(pack['train'],data.labels[train[a]],c);p=model.predict_proba(pack['test'])[:,1];pred[(f,c)][b]=p
                aucs[(f,c)].append(float(roc_auc_score(data.labels[train[b]],p)));aps[(f,c)].append(float(average_precision_score(data.labels[train[b]],p)))
    assert (seen==1).all() and all(np.isfinite(p).all() for p in pred.values())
    rows=[dict(family=f,C=c,mean_inner_AUROC=float(np.mean(aucs[(f,c)])),mean_inner_AP=float(np.mean(aps[(f,c)])),inner_AUROCs=';'.join(map(str,aucs[(f,c)])),inner_APs=';'.join(map(str,aps[(f,c)])),inner_eligible_pathways=';'.join(map(str,counts[f]))) for f,c in keys];best={}
    for f in PATH_FAMILIES:
        c=min([r for r in rows if r['family']==f],key=lambda r:(-r['mean_inner_AUROC'],r['C'])).copy();threshold,ba=select_threshold(data.labels[train],pred[(f,c['C'])]);c.update(threshold=threshold,inner_threshold_balanced_accuracy=ba);best[f]=c
    return best,rows,pred,pd.concat(eligibility,ignore_index=True)

def artifact(model,pack,chosen,genes):
    return dict(classifier=model,family=chosen['family'],hyperparameters=chosen,threshold=chosen['threshold'],positive_class='PD',feature_universe=genes,pathway_state=pack['state'],method='non-directional singscore arithmetic; min ties; fixed training background; counts/featureCounts length')

def predict_artifact(state,raw):
    raw=np.asarray(raw);assert raw.ndim==2 and raw.shape[1]==len(state['feature_universe']) and np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
    s=state['pathway_state'];v=cpu_scores(raw[:,s['background_indices']],s['lengths'],s['membership']) if len(s['pathway_indices']) else np.empty((len(raw),0))
    return state['classifier'].predict_proba((v-s['means'])/s['scales'])[:,1]
