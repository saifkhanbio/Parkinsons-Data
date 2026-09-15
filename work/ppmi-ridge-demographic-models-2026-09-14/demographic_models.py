"""Controlled addition of age and sex to the original Hallmark ridge."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
MAPPED = HERE.parent / "ppmi-mapped-gene-classifier-2026-09-12"
sys.path.insert(0, str(MAPPED))
from mapped_genes import (BASE, PRIOR, HALLMARK, MappedData, choose_device,
    make_splits, SEED, fit_model, select_threshold, evaluate)

FAMILIES = ["rna", "demographic", "rna_demographic"]
CS = [.001, .01, .1, 1.]
DEMOGRAPHIC_FEATURES = ["age_collection_years", "male"]


def demographic_values(metadata):
    assert metadata.sex.isin(["Male", "Female"]).all()
    result = np.column_stack([metadata.age_collection_years.to_numpy(float),
                              metadata.sex.eq("Male").to_numpy(float)])
    assert np.isfinite(result).all()
    return result


class DemographicData(MappedData):
    def __init__(self, raw, genes, metadata, labels, indices, device):
        super().__init__(raw, genes, labels, indices, device)
        self.demographics = demographic_values(metadata)

    def pack(self, train, test):
        result = super().pack(train, test)
        mean = self.demographics[train].mean(0)
        scale = self.demographics[train].std(0)
        scale[scale < 1e-12] = 1.
        result["demographic"] = {
            "train": (self.demographics[train]-mean)/scale,
            "test": (self.demographics[test]-mean)/scale,
            "means": mean, "scales": scale, "features": DEMOGRAPHIC_FEATURES}
        return result


def matrices(pack, family):
    if family == "rna":
        return pack["train"], pack["test"]
    d = pack["demographic"]
    if family == "demographic":
        return d["train"], d["test"]
    assert family == "rna_demographic"
    return np.column_stack([d["train"], pack["train"]]), np.column_stack([d["test"], pack["test"]])


def tune(data, train, splits, cs=CS):
    keys = [(f,c) for f in FAMILIES for c in cs]
    predictions = {key:np.full(len(train),np.nan) for key in keys}
    scores = {key:[] for key in keys}; aps = {key:[] for key in keys}
    iterations = {key:[] for key in keys}; seen=np.zeros(len(train),dtype=int)
    counts=[]
    for a,b in splits:
        assert not np.intersect1d(a,b).size
        seen[b]+=1
        pack=data.pack(train[a],train[b]);counts.append(pack["train"].shape[1])
        for family in FAMILIES:
            x,v=matrices(pack,family)
            for c in cs:
                key=(family,c)
                model=fit_model(x,data.labels[train[a]],c,"ridge",0.)
                p=model.predict_proba(v)[:,1]
                assert np.isfinite(p).all()
                predictions[key][b]=p
                scores[key].append(float(roc_auc_score(data.labels[train[b]],p)))
                aps[key].append(float(average_precision_score(data.labels[train[b]],p)))
                iterations[key].append(int(model.n_iter_[0]))
    assert (seen==1).all()
    rows=[];best={}
    for family,c in keys:
        key=(family,c)
        assert np.isfinite(predictions[key]).all()
        rows.append(dict(family=family,C=c,mean_inner_AUROC=float(np.mean(scores[key])),
                         mean_inner_AP=float(np.mean(aps[key])),inner_AUROCs=";".join(map(str,scores[key])),
                         inner_APs=";".join(map(str,aps[key])),inner_eligible_counts=";".join(map(str,counts)),
                         maximum_optimizer_iterations=max(iterations[key])))
    for family in FAMILIES:
        chosen=min([r for r in rows if r["family"]==family],key=lambda r:(-r["mean_inner_AUROC"],r["C"])).copy()
        threshold,ba=select_threshold(data.labels[train],predictions[(family,chosen["C"])])
        chosen.update(threshold=threshold,inner_threshold_balanced_accuracy=ba)
        best[family]=chosen
    return best,rows,predictions


def artifact(model, pack, chosen, genes):
    family=chosen["family"]
    state=dict(classifier=model,family=family,hyperparameters=chosen,threshold=chosen["threshold"],
               positive_class="PD",feature_universe=np.asarray(genes),training_transform="log2(1+CPM), training mean/SD")
    if family != "demographic":
        state.update(rna=pack["state"],selected_gene_ids=np.asarray(genes)[pack["state"]["indices"]])
    if family != "rna":
        d=pack["demographic"]
        state["demographic"]={k:d[k] for k in ["means","scales","features"]}
    return state


def predict_artifact(state, raw, metadata):
    parts=[]
    if state["family"]!="rna":
        d=state["demographic"]
        parts.append((demographic_values(metadata)-d["means"])/d["scales"])
    if state["family"]!="demographic":
        raw=np.asarray(raw)
        assert raw.ndim==2 and raw.shape[1]==len(state["feature_universe"])
        assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
        lib=raw.sum(1,dtype=np.float64);assert (lib>0).all()
        r=state["rna"]
        parts.append((np.log2(1+raw[:,r["indices"]]/lib[:,None]*1e6)-r["means"])/r["scales"])
    return state["classifier"].predict_proba(np.column_stack(parts))[:,1]


def preflight(device):
    import io,joblib,torch
    rng=np.random.default_rng(SEED)
    raw=rng.poisson(30,(80,250)).astype(np.int32);raw[:,:2]=0
    y=np.tile([0,1],40);raw[y==1,5:10]+=12
    meta=pd.DataFrame({"age_collection_years":rng.uniform(35,80,80),"sex":np.where(rng.random(80)>.5,"Male","Female")})
    genes=np.arange(250).astype(str);indices=np.arange(180)
    tr,te=np.arange(60),np.arange(60,80)
    splits=make_splits(y[tr],np.arange(60),"participant_stratified",3,SEED)
    data=DemographicData(raw,genes,meta,y,indices,device)
    pack=data.pack(tr,te);best,rows,_=tune(data,tr,splits,cs=[.01,.1])
    changed_raw=raw.copy();changed_raw[te]+=10000
    changed_y=y.copy();changed_y[te]=1-changed_y[te]
    changed_meta=meta.copy();changed_meta.loc[te,"age_collection_years"]+=1000
    changed_meta.loc[te,"sex"]=np.where(meta.loc[te,"sex"]=="Male","Female","Male")
    changed=DemographicData(changed_raw,genes,changed_meta,changed_y,indices,device)
    pack2=changed.pack(tr,te);best2,rows2,_=tune(changed,tr,splits,cs=[.01,.1])
    assert best==best2 and rows==rows2
    for family in FAMILIES:
        np.testing.assert_array_equal(matrices(pack,family)[0],matrices(pack2,family)[0])
        x,v=matrices(pack,family)
        model=fit_model(x,y[tr],best[family]["C"],"ridge",0.)
        state=artifact(model,pack,best[family],genes)
        buf=io.BytesIO();joblib.dump(state,buf);buf.seek(0);restored=joblib.load(buf)
        np.testing.assert_allclose(predict_artifact(restored,raw[te],meta.iloc[te]),model.predict_proba(v)[:,1],atol=1e-9,rtol=1e-9)
    reversed_data=DemographicData(raw,genes,meta,1-y,indices,device).pack(tr,te)
    for family in FAMILIES:
        np.testing.assert_array_equal(matrices(pack,family)[0],matrices(reversed_data,family)[0])
    constant=meta.copy();constant.loc[tr,"sex"]="Male"
    cc=DemographicData(raw,genes,constant,y,indices,device).pack(tr,te)["demographic"]
    assert cc["scales"][1]==1 and np.all(cc["train"][:,1]==0) and np.isfinite(cc["test"]).all()
    cpu=DemographicData(raw,genes,meta,y,indices,torch.device("cpu")).pack(tr,te)
    for family in FAMILIES:
        np.testing.assert_allclose(matrices(pack,family)[0],matrices(cpu,family)[0],atol=1e-9,rtol=1e-9)
    return {"heldout_RNA_age_sex_label_perturbation":"PASS","training_only_scaling":"PASS",
            "label_independent_preprocessing":"PASS","constant_demographic_column":"PASS",
            "CPU_device_agreement":"PASS","artifact_serialization_and_inference":"PASS"}
