"""Hallmark RNA and measured-cell blocks with independently tuned L2 penalties."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
HERE=Path(__file__).resolve().parent;MAPPED=HERE.parent/'ppmi-mapped-gene-classifier-2026-09-12'
sys.path.insert(0,str(MAPPED))
from mapped_genes import BASE,PRIOR,HALLMARK,MappedData,choose_device,make_splits,SEED,fit_model,select_threshold,evaluate
FAMILIES=['rna','blood','rna_blood'];CS=[.001,.01,.1,1.]
BLOOD_FEATURES=['log_wbc','neutrophils_percent','monocytes_percent','eosinophils_percent','basophils_percent']

def blood_values(metadata):
    wbc=metadata.wbc.to_numpy(float);p=metadata[BLOOD_FEATURES[1:]].to_numpy(float)
    assert np.isfinite(wbc).all() and (wbc>0).all()
    assert np.isfinite(p).all() and ((p>=0)&(p<=100)).all()
    return np.column_stack([np.log(wbc),p])

class BloodData(MappedData):
    def __init__(self,raw,genes,metadata,labels,indices,device):
        super().__init__(raw,genes,labels,indices,device);self.blood=blood_values(metadata)
    def pack(self,train,test):
        result=super().pack(train,test);mean=self.blood[train].mean(0);scale=self.blood[train].std(0);scale[scale<1e-12]=1.
        result['blood']=dict(train=(self.blood[train]-mean)/scale,test=(self.blood[test]-mean)/scale,means=mean,scales=scale,features=BLOOD_FEATURES)
        return result

def grid(cs=CS):
    return ([dict(family='rna',C=c,C_RNA=c,C_blood=None) for c in cs]+
            [dict(family='blood',C=c,C_RNA=None,C_blood=c) for c in cs]+
            [dict(family='rna_blood',C=a,C_RNA=a,C_blood=b) for a in cs for b in cs])

def key(c):return c['family'],c['C_RNA'],c['C_blood']

def multiplier(c):return np.sqrt(c['C_blood']/c['C_RNA']) if c['family']=='rna_blood' else 1.

def matrices(pack,config):
    family=config['family']
    if family=='rna':return pack['train'],pack['test']
    b=pack['blood']
    if family=='blood':return b['train'],b['test']
    assert family=='rna_blood';m=multiplier(config)
    return np.column_stack([b['train']*m,pack['train']]),np.column_stack([b['test']*m,pack['test']])

def tune(data,train,splits,cs=CS):
    candidates=grid(cs);pred={key(c):np.full(len(train),np.nan) for c in candidates}
    scores={key(c):[] for c in candidates};aps={key(c):[] for c in candidates};iterations={key(c):[] for c in candidates}
    counts=[];seen=np.zeros(len(train),int)
    for a,b in splits:
        seen[b]+=1;pack=data.pack(train[a],train[b]);counts.append(pack['train'].shape[1])
        for c in candidates:
            k=key(c);x,v=matrices(pack,c);model=fit_model(x,data.labels[train[a]],c['C'],'ridge',0.)
            p=model.predict_proba(v)[:,1];pred[k][b]=p;scores[k].append(float(roc_auc_score(data.labels[train[b]],p)));aps[k].append(float(average_precision_score(data.labels[train[b]],p)));iterations[k].append(int(model.n_iter_[0]))
    assert (seen==1).all() and all(np.isfinite(p).all() for p in pred.values())
    rows=[dict(**c,mean_inner_AUROC=float(np.mean(scores[key(c)])),mean_inner_AP=float(np.mean(aps[key(c)])),inner_AUROCs=';'.join(map(str,scores[key(c)])),inner_APs=';'.join(map(str,aps[key(c)])),inner_eligible_counts=';'.join(map(str,counts)),maximum_optimizer_iterations=max(iterations[key(c)])) for c in candidates]
    best={}
    for family in FAMILIES:
        c=min([r for r in rows if r['family']==family],key=lambda r:(-r['mean_inner_AUROC'],r['C_RNA'] or 0,r['C_blood'] or 0)).copy()
        threshold,ba=select_threshold(data.labels[train],pred[key(c)]);c.update(threshold=threshold,inner_threshold_balanced_accuracy=ba);best[family]=c
    return best,rows,pred

def artifact(model,pack,chosen,genes):
    family=chosen['family'];s=dict(classifier=model,family=family,hyperparameters=chosen,threshold=chosen['threshold'],positive_class='PD',feature_universe=np.asarray(genes),blood_multiplier=float(multiplier(chosen)),training_transform='log2(1+CPM), training mean/SD; blood log-WBC and percentages')
    if family!='blood':s.update(rna=pack['state'],selected_gene_ids=np.asarray(genes)[pack['state']['indices']])
    if family!='rna':s['blood']={k:pack['blood'][k] for k in ['means','scales','features']}
    return s

def predict_artifact(state,raw,metadata):
    parts=[]
    if state['family']!='rna':
        b=state['blood'];parts.append((blood_values(metadata)-b['means'])/b['scales']*state['blood_multiplier'])
    if state['family']!='blood':
        raw=np.asarray(raw);assert raw.ndim==2 and raw.shape[1]==len(state['feature_universe'])
        assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
        lib=raw.sum(1,dtype=np.float64);assert (lib>0).all();r=state['rna']
        parts.append((np.log2(1+raw[:,r['indices']]/lib[:,None]*1e6)-r['means'])/r['scales'])
    return state['classifier'].predict_proba(np.column_stack(parts))[:,1]

def penalty_check():
    from scipy.optimize import minimize
    from scipy.special import expit
    rng=np.random.default_rng(774);x=rng.normal(size=(120,8));y=(x[:,0]+.5*x[:,5]+rng.normal(size=120)>.1).astype(int)
    worst=0.
    for cr,cb in [(.001,1.),(.1,.1),(1.,.001)]:
        mult=np.sqrt(cb/cr);scaled=x.copy();scaled[:,:3]*=mult;model=fit_model(scaled,y,cr,'ridge',0.)
        coef=model.coef_[0].copy();coef[:3]*=mult;theta=np.r_[coef,model.intercept_[0]]
        penalty=np.r_[np.repeat(1/(len(y)*cb),3),np.repeat(1/(len(y)*cr),5),0.]
        design=np.column_stack([x,np.ones(len(x))])
        def loss(v):
            z=design@v;res=expit(z)-y
            return np.mean(np.logaddexp(0,z)-y*z)+.5*np.sum(penalty*v*v),design.T@res/len(y)+penalty*v
        ref=minimize(loss,np.zeros(9),jac=True,method='L-BFGS-B',options={'gtol':1e-11,'ftol':1e-15,'maxiter':10000})
        assert np.max(abs(loss(ref.x)[1]))<1e-7
        err=float(np.max(abs(expit(design@theta)-expit(design@ref.x))));worst=max(worst,err)
        assert err<1e-4 and np.max(abs(loss(theta)[1]))<1e-5
        if cr==cb:
            ordinary=fit_model(x,y,cr,'ridge',0.);np.testing.assert_allclose(model.predict_proba(scaled),ordinary.predict_proba(x),atol=1e-12)
    return worst

def preflight(device):
    import io,joblib,torch
    rng=np.random.default_rng(SEED);raw=rng.poisson(30,(80,250)).astype(np.int32);raw[:,:2]=0;y=np.tile([0,1],40);raw[y==1,5:10]+=12
    meta=pd.DataFrame(dict(wbc=rng.uniform(4,9,80),neutrophils_percent=rng.uniform(30,60,80),monocytes_percent=rng.uniform(2,10,80),eosinophils_percent=rng.uniform(0,5,80),basophils_percent=rng.uniform(0,1,80)))
    genes=np.arange(250).astype(str);indices=np.arange(180);tr=np.arange(60);te=np.arange(60,80)
    splits=make_splits(y[tr],np.arange(60),'participant_stratified',3,SEED)
    data=BloodData(raw,genes,meta,y,indices,device);pack=data.pack(tr,te);best,rows,_=tune(data,tr,splits,cs=[.01,.1])
    changed_raw=raw.copy();changed_raw[te]+=10000;changed_y=y.copy();changed_y[te]=1-y[te]
    changed_meta=meta.copy();changed_meta.loc[te,'wbc']*=2;changed_meta.loc[te,BLOOD_FEATURES[1:]]=1.
    other=BloodData(changed_raw,genes,changed_meta,changed_y,indices,device);b2,r2,_=tune(other,tr,splits,cs=[.01,.1]);assert best==b2 and rows==r2
    cpu=BloodData(raw,genes,meta,y,indices,torch.device('cpu')).pack(tr,te)
    reversed_pack=BloodData(raw,genes,meta,1-y,indices,device).pack(tr,te)
    for family in FAMILIES:
        c=best[family];x,v=matrices(pack,c);np.testing.assert_allclose(x,matrices(cpu,c)[0],atol=1e-9,rtol=1e-9)
        np.testing.assert_array_equal(x,matrices(reversed_pack,c)[0]);np.testing.assert_array_equal(x,matrices(other.pack(tr,te),c)[0])
        model=fit_model(x,y[tr],c['C'],'ridge',0.);s=artifact(model,pack,c,genes);buf=io.BytesIO();joblib.dump(s,buf);buf.seek(0)
        np.testing.assert_allclose(predict_artifact(joblib.load(buf),raw[te],meta.iloc[te]),model.predict_proba(v)[:,1],atol=1e-9,rtol=1e-9)
    constant=meta.copy();constant.loc[tr,'basophils_percent']=0.;d=BloodData(raw,genes,constant,y,indices,device).pack(tr,te)['blood']
    assert d['scales'][-1]==1 and (d['train'][:,-1]==0).all()
    return dict(heldout_RNA_cell_label_perturbation='PASS',training_only_scaling='PASS',label_independent_preprocessing='PASS',constant_cell_column='PASS',CPU_GPU_agreement=('PASS' if device.type=='cuda' else 'Not run: CPU-only preflight'),serialized_inference='PASS',independent_penalty_reference_max_probability_error=penalty_check())
