"""Tie-aware AUROC and average precision with bootstrap multiplicities."""
import numpy as np
import torch

def weighted_metrics(y,scores,weights,device):
    order=np.argsort(-scores,kind="stable")
    ends=np.r_[np.flatnonzero(np.diff(scores[order])!=0),len(y)-1]
    labels=torch.as_tensor(y[order],dtype=torch.float64,device=device)
    endpoints=torch.as_tensor(ends,dtype=torch.long,device=device)
    result=[]
    for start in range(0,len(weights),512):
        w=torch.as_tensor(weights[start:start+512,order],dtype=torch.float64,device=device)
        tp=torch.cumsum(w*labels,dim=1)[:,endpoints]
        fp=torch.cumsum(w*(1-labels),dim=1)[:,endpoints]
        dtp=torch.diff(tp,prepend=torch.zeros_like(tp[:,:1]),dim=1)
        dfp=torch.diff(fp,prepend=torch.zeros_like(fp[:,:1]),dim=1)
        positive,negative=tp[:,-1],fp[:,-1]
        auc=torch.sum(dfp*(tp-.5*dtp),dim=1)/(positive*negative)
        precision=torch.where(tp+fp>0,tp/(tp+fp),0.)
        ap=torch.sum(dtp*precision,dim=1)/positive
        v=torch.stack([auc,ap],dim=1)
        v[(positive==0)|(negative==0)]=float("nan")
        result.append(v.cpu().numpy())
    return np.concatenate(result)

