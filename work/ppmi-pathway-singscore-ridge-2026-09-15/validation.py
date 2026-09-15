"""Independent R formula checks, training isolation, and inference tests."""
import io,subprocess
import numpy as np
import pandas as pd
import joblib,torch
from scipy import sparse
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from path_models import ROOT,PathData,PATH_FAMILIES,score_tensor,cpu_scores,tune,fit,artifact,predict_artifact

def preflight(device):
    rng=np.random.default_rng(883);worst=0.
    for n in [45,46]:
        raw=rng.integers(0,20,(7,n));raw[0]=0;length=rng.integers(1,6,n).astype(float)
        sets=[np.arange(15),np.arange(5,21),np.arange(10,31)]
        rows=np.concatenate([np.repeat(i,len(s)) for i,s in enumerate(sets)]);cols=np.concatenate(sets)
        m=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(3,n))
        x=raw/length;pd.DataFrame(x).to_csv(ROOT/'tmp/reference_expression.tsv',sep='\t',index=False,float_format='%.17g')
        pd.DataFrame(dict(pathway=rows,gene_index=cols)).to_csv(ROOT/'tmp/reference_members.tsv',sep='\t',index=False)
        subprocess.run(['Rscript',str(ROOT/'reference_scores.R'),str(ROOT/'tmp/reference_expression.tsv'),str(ROOT/'tmp/reference_members.tsv'),str(ROOT/'tmp/reference_output.tsv')],check=True,capture_output=True,text=True)
        reference=np.loadtxt(ROOT/'tmp/reference_output.tsv');actual=score_tensor(torch.tensor(x,device=device,dtype=torch.float64),m,device).cpu().numpy()
        np.testing.assert_allclose(actual,reference,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(actual,cpu_scores(raw,length,m),atol=1e-12,rtol=1e-12)
        worst=max(worst,float(np.max(abs(actual-reference))))
        denominator=x.sum(1);valid=denominator>0
        np.testing.assert_array_equal(rankdata(x[valid],axis=1,method='min'),rankdata(x[valid]/denominator[valid,None]*1e6,axis=1,method='min'))
    raw=rng.poisson(30,(60,620)).astype(np.int32);y=np.tile([0,1],30);raw[y==1,5:15]+=15
    genes=np.arange(620).astype(str);length=rng.integers(100,9000,620).astype(float)
    ref=pd.DataFrame(dict(pathway_index=np.arange(18),pathway=['set'+str(i) for i in range(18)],collection=np.repeat(['Hallmark','C2_CP','C5_BP'],6)))
    sets=[rng.choice(620,30,replace=False) for _ in range(18)]
    matrix=sparse.csr_matrix((np.ones(18*30),(np.repeat(np.arange(18),30),np.concatenate(sets))),shape=(18,620))
    tr=np.arange(45);te=np.arange(45,60);splits=list(StratifiedKFold(3,shuffle=True,random_state=552).split(tr,y[tr]))
    data=PathData(raw,genes,y,length,np.ones(620,bool),matrix,ref,device);best,rows,_,_=tune(data,tr,splits)
    altered=raw.copy();altered[te]*=13;other_y=y.copy();other_y[te]=1-y[te]
    other=PathData(altered,genes,other_y,length,np.ones(620,bool),matrix,ref,device);b2,r2,_,_=tune(other,tr,splits);assert best==b2 and rows==r2
    packs,_=data.pack(tr,te);packs2,_=other.pack(tr,te)
    for family in PATH_FAMILIES:
        a=packs[family];np.testing.assert_array_equal(a['train'],packs2[family]['train'])
        model=fit(a['train'],y[tr],best[family]['C']);s=artifact(model,a,best[family],genes)
        buf=io.BytesIO();joblib.dump(s,buf);buf.seek(0);s=joblib.load(buf)
        predicted=predict_artifact(s,raw[te]);np.testing.assert_allclose(predicted,model.predict_proba(a['test'])[:,1],atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(predicted,np.concatenate([predict_artifact(s,raw[[i]]) for i in te]),atol=1e-12,rtol=0)
    empty=fit(np.empty((len(tr),0)),y[tr],.001);assert np.all(empty.predict_proba(np.empty((len(te),0)))[:,1]==y[tr].mean())
    return dict(reference_R_formula='PASS',reference_max_error=worst,ties_even_odd_backgrounds='PASS',TPM_rank_equivalence='PASS',heldout_RNA_label_perturbation='PASS',training_background_and_scaling='PASS',serialized_inference='PASS',single_sample_invariance='PASS',zero_feature_fallback='PASS',device=str(device))
