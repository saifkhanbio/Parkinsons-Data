"""Partition-specific, outcome-blind removal of one technical expression component."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent;MAPPED=ROOT.parent/'ppmi-mapped-gene-classifier-2026-09-12'
sys.path.insert(0,str(MAPPED))
from mapped_genes import BASE,HALLMARK,PRIOR,MappedData,choose_device,make_splits,fit_model,select_threshold,evaluate
FAMILIES=['rna','intergenic_adjusted'];CS=[.001,.01,.1,1.]
class ConstantModel:
    def __init__(self,y):self.p=float(np.mean(y));self.coef_=np.empty((1,0));self.n_iter_=np.array([0])
    def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))
def fit(x,y,c):return fit_model(x,y,c,'ridge',0.) if x.shape[1] else ConstantModel(y)

def residualize(a,b,train_qc,test_qc,device):
    a=torch.as_tensor(a,dtype=torch.float64,device=device);b=torch.as_tensor(b,dtype=torch.float64,device=device)
    mean=float(np.mean(train_qc));sd=float(np.std(train_qc));active=sd>1e-12
    if active:
        z=torch.tensor((train_qc-mean)/sd,dtype=torch.float64,device=device);v=torch.tensor((test_qc-mean)/sd,dtype=torch.float64,device=device)
        beta=z@a/(z@z);ar=a-z[:,None]*beta;bt=b-v[:,None]*beta
        mu=ar.mean(0);scale=ar.std(0,unbiased=False);keep=scale>1e-12
        x=(ar[:,keep]-mu[keep])/scale[keep];t=(bt[:,keep]-mu[keep])/scale[keep]
        orth=float(torch.max(torch.abs(z@x/len(z))).cpu()) if x.shape[1] else 0.
        assert orth<1e-9
    else:
        beta=torch.zeros(a.shape[1],dtype=torch.float64,device=device);mu=beta.clone();scale=torch.ones_like(beta);keep=torch.ones(a.shape[1],dtype=torch.bool,device=device);x=a;t=b;orth=0.
    state=dict(qc_feature='intergenic_percent',qc_mean=mean,qc_SD=sd,qc_active=active,beta=beta.cpu().numpy(),residual_means=mu.cpu().numpy(),residual_scales=scale.cpu().numpy(),residual_keep=keep.cpu().numpy(),training_orthogonality_max=orth,training_qc_min=float(np.min(train_qc)),training_qc_max=float(np.max(train_qc)))
    return dict(train=x.cpu().numpy(),test=t.cpu().numpy(),adjustment=state)

class AdjustData(MappedData):
    def __init__(self,raw,genes,y,indices,qc,device):
        super().__init__(raw,genes,y,indices,device);self.qc=np.asarray(qc,float);self.device=device
        assert self.qc.shape==(len(y),) and np.isfinite(self.qc).all() and ((self.qc>=0)&(self.qc<=100)).all()
    def pack(self,tr,te):
        base=super().pack(tr,te);adj=residualize(base['train'],base['test'],self.qc[tr],self.qc[te],self.device);adj['state']=base['state']
        return {'rna':dict(**base,adjustment=None),'intergenic_adjusted':adj}

def tune(data,tr,splits,cs=CS):
    preds={(f,c):np.full(len(tr),np.nan) for f in FAMILIES for c in cs};scores={k:[] for k in preds};aps={k:[] for k in preds};counts={f:[] for f in FAMILIES};seen=np.zeros(len(tr),int)
    for a,b in splits:
        seen[b]+=1;pack=data.pack(tr[a],tr[b])
        for f in FAMILIES:
            x=pack[f];counts[f].append(x['train'].shape[1])
            for c in cs:
                model=fit(x['train'],data.labels[tr[a]],c);p=model.predict_proba(x['test'])[:,1];preds[(f,c)][b]=p
                scores[(f,c)].append(roc_auc_score(data.labels[tr[b]],p));aps[(f,c)].append(average_precision_score(data.labels[tr[b]],p))
    assert (seen==1).all() and all(np.isfinite(p).all() for p in preds.values())
    rows=[dict(family=f,C=c,mean_inner_AUROC=float(np.mean(scores[(f,c)])),mean_inner_AP=float(np.mean(aps[(f,c)])),inner_AUROCs=';'.join(map(str,scores[(f,c)])),inner_feature_counts=';'.join(map(str,counts[f]))) for f,c in preds]
    best={}
    for f in FAMILIES:
        r=min([r for r in rows if r['family']==f],key=lambda r:(-r['mean_inner_AUROC'],r['C'])).copy();th,ba=select_threshold(data.labels[tr],preds[(f,r['C'])]);r.update(threshold=th,inner_threshold_balanced_accuracy=ba);best[f]=r
    return best,rows,preds

def artifact(model,pack,chosen,genes):
    return dict(classifier=model,family=chosen['family'],hyperparameters=chosen,threshold=chosen['threshold'],feature_universe=np.asarray(genes),rna=pack['state'],adjustment=pack['adjustment'],positive_class='PD')

def predict(state,raw,qc=None):
    raw=np.asarray(raw);assert raw.ndim==2 and raw.shape[1]==len(state['feature_universe']) and np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
    lib=raw.sum(1,dtype=float);assert (lib>0).all();s=state['rna'];x=(np.log2(1+raw[:,s['indices']]/lib[:,None]*1e6)-s['means'])/s['scales']
    a=state['adjustment']
    if a is not None:
        qc=np.asarray(qc,float);assert qc.shape==(len(raw),) and np.isfinite(qc).all() and ((qc>=0)&(qc<=100)).all()
        z=(qc-a['qc_mean'])/a['qc_SD'] if a['qc_active'] else np.zeros(len(raw))
        keep=a['residual_keep'];x=(x-z[:,None]*a['beta']-a['residual_means'])[:,keep]/a['residual_scales'][keep]
    return state['classifier'].predict_proba(x)[:,1]

def preflight(device):
    import pickle
    rng=np.random.default_rng(867);qc=rng.uniform(5,25,80);raw=rng.poisson(30,(80,250)).astype(np.int32);raw[:,:2]=0
    raw[:,3:12]+=(qc[:,None]*2).astype(int);y=np.tile([0,1],40);genes=np.arange(250).astype(str);indices=np.arange(180);tr=np.arange(60);te=np.arange(60,80)
    data=AdjustData(raw,genes,y,indices,qc,device);packs=data.pack(tr,te);base=packs['rna'];adj=packs['intergenic_adjusted'];a=adj['adjustment']
    z=(qc[tr]-qc[tr].mean())/qc[tr].std();coeff=np.linalg.lstsq(np.c_[np.ones(len(tr)),z],base['train'],rcond=None)[0]
    np.testing.assert_allclose(a['beta'],coeff[1],atol=1e-10)
    r=base['train']-z[:,None]*coeff[1];np.testing.assert_allclose(adj['train'],(r-r.mean(0))/r.std(0),atol=1e-10)
    cpu=residualize(base['train'],base['test'],qc[tr],qc[te],torch.device('cpu'));np.testing.assert_allclose(cpu['test'],adj['test'],atol=1e-10)
    splits=make_splits(y[tr],np.arange(len(tr)),'participant_stratified',3,911);best,_,_=tune(data,tr,splits,cs=[.01,.1])
    raw2=raw.copy();raw2[te]+=10000;qc2=qc.copy();qc2[te]=98.;y2=y.copy();y2[te]=1-y2[te]
    other=AdjustData(raw2,genes,y2,indices,qc2,device);b=other.pack(tr,te);best2,_,_=tune(other,tr,splits,cs=[.01,.1]);assert best==best2
    np.testing.assert_array_equal(adj['train'],b['intergenic_adjusted']['train'])
    swapped=AdjustData(raw,genes,1-y,indices,qc,device);np.testing.assert_array_equal(adj['train'],swapped.pack(tr,te)['intergenic_adjusted']['train'])
    const=residualize(base['train'],base['test'],np.ones(len(tr)),qc[te],device);np.testing.assert_array_equal(const['train'],base['train']);np.testing.assert_array_equal(const['test'],base['test'])
    linear=residualize(z[:,None]*np.array([[1.,2.]]),np.ones((len(te),2)),qc[tr],qc[te],device);assert linear['train'].shape[1]==0
    cm=fit(linear['train'],y[tr],.01);assert np.all(cm.predict_proba(linear['test'])[:,1]==y[tr].mean())
    for f in FAMILIES:
        model=fit(packs[f]['train'],y[tr],best[f]['C']);s=artifact(model,packs[f],best[f],genes);s=pickle.loads(pickle.dumps(s))
        np.testing.assert_allclose(predict(s,raw[te],qc[te]),model.predict_proba(packs[f]['test'])[:,1],atol=1e-9)
    return dict(status='PASS',independent_OLS='PASS',CPU_GPU='PASS',heldout_RNA_QC_label_perturbation='PASS',label_independent_preprocessing='PASS',constant_QC='PASS',zero_residual_features='PASS',artifact_inference='PASS',device=str(device))
