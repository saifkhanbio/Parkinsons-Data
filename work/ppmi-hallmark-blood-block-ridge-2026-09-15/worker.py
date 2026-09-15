"""Nested Hallmark plus measured-cell ridge comparison with separate penalties."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[key]="2"
for folder in ["tmp","cache","matplotlib_cache"]:(HERE/folder).mkdir(exist_ok=True)
os.environ.update(MPLCONFIGDIR=str(HERE/"matplotlib_cache"),TMPDIR=str(HERE/"tmp"),XDG_CACHE_HOME=str(HERE/"cache"),PYTHONDONTWRITEBYTECODE="1")
import sys,json,hashlib,traceback,time
sys.dont_write_bytecode=True
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch,joblib,sklearn
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score,average_precision_score
from blood_models import (BASE,PRIOR,HALLMARK,MAPPED,FAMILIES,CS,BloodData,
    choose_device,tune,matrices,fit_model,artifact,predict_artifact,evaluate,preflight)

KEYS=["repeat","protocol","fold"]


def save(name,value):
    path=HERE/name;tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(path)


def status(state,**kw):
    save("status.json",dict(status=state,utc=datetime.now(timezone.utc).isoformat(),**kw))


def table(name,rows):
    pd.DataFrame(rows).to_csv(HERE/name,sep="\t",index=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def prepare(device):
    status("PREPARING")
    inputs=[BASE/n for n in ["counts.npy","gene_ids.npy","metadata.tsv","classifier.py"]]
    inputs += [HALLMARK/"membership.npz",PRIOR/"refine.py",PRIOR/"elastic_solver.py"]
    inputs += [MAPPED/n for n in ["mapped_genes.py","fold_plan.json","candidate_genes.tsv", "input_sha256.json",
        "out_of_fold_predictions.tsv","selected_parameters.tsv","inner_tuning_scores.tsv",
        "outer_fold_metrics.tsv","outer_gene_coefficients.tsv","primary_summary.tsv","final_tuning_folds.json"]]
    old_hashes=json.loads((MAPPED/"input_sha256.json").read_text())
    for path in inputs:
        assert path.exists(),path
        if str(path) in old_hashes:assert sha(path)==old_hashes[str(path)],path
    raw=np.load(BASE/"counts.npy");genes=np.load(BASE/"gene_ids.npy")
    meta=pd.read_csv(BASE/"metadata.tsv",sep="\t",dtype={"PATNO":str});y=meta.label.to_numpy()
    assert raw.shape==(528,58780) and len(genes)==len(np.unique(genes))==58780
    assert np.issubdtype(raw.dtype,np.integer) and (raw>=0).all()
    assert len(meta)==528 and meta.PATNO.is_unique
    assert meta.group.value_counts().to_dict()=={"PD":358,"Control":170}
    assert np.array_equal(y,meta.group.eq("PD").astype(int))
    indices=np.load(HALLMARK/"membership.npz")["indices"]
    assert len(indices)==len(np.unique(indices))==4376
    candidates=pd.read_csv(MAPPED/"candidate_genes.tsv",sep="\t")
    np.testing.assert_array_equal(indices,candidates.raw_index)
    np.testing.assert_array_equal(genes[indices],candidates.Geneid)
    table("candidate_genes.tsv",candidates)
    table("metadata.tsv",meta)
    from blood_models import grid
    save("candidate_grid.json",grid())
    plan=json.loads((MAPPED/"fold_plan.json").read_text());assert len(plan)==30
    coverage={};batches=meta.batch.to_numpy(str)
    for split in plan:
        tr,te=np.array(split["train_indices"]),np.array(split["test_indices"])
        assert len(np.unique(tr))==len(tr) and len(np.unique(te))==len(te)
        assert not np.intersect1d(tr,te).size and set(np.r_[tr,te])==set(range(528))
        assert len(np.unique(y[tr]))==len(np.unique(y[te]))==2
        key=(split["repeat"],split["protocol"])
        coverage.setdefault(key,np.zeros(528,dtype=int))[te]+=1
        if split["protocol"]=="batch_grouped":assert not set(batches[tr])&set(batches[te])
        seen=np.zeros(len(tr),dtype=int)
        for inner in split["inner"]:
            a,b=np.array(inner["train_positions"]),np.array(inner["validation_positions"])
            assert not np.intersect1d(a,b).size and set(np.r_[a,b])==set(range(len(tr)))
            assert len(np.unique(y[tr[a]]))==len(np.unique(y[tr[b]]))==2
            if split["protocol"]=="batch_grouped":assert not set(batches[tr[a]])&set(batches[tr[b]])
            seen[b]+=1
        assert (seen==1).all()
    assert len(coverage)==6 and all((v==1).all() for v in coverage.values())
    save("fold_plan.json",plan)
    assert meta.CBC_month_gap.isin([-2,-1,0]).all()
    checks=preflight(device)
    inputs += [HERE/n for n in ["README.md","blood_models.py","worker.py","launch.py","reporting.py","predict.py"]]
    save("input_sha256.json",{str(p):sha(p) for p in inputs})
    save("preflight.json",dict(status="PASS",checks=checks,participants=528,outer_folds=30,
         device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
         sklearn=sklearn.__version__,torch=torch.__version__,python=sys.executable))
    return raw,genes,meta,indices,plan


def train(device):
    started=time.monotonic()
    raw,genes,meta,indices,plan=prepare(device);y=meta.label.to_numpy()
    data=BloodData(raw,genes,meta,y,indices,device)
    old_params=pd.read_csv(MAPPED/"selected_parameters.tsv",sep="\t").query("penalty=='ridge'").set_index(KEYS)
    old_inner=pd.read_csv(MAPPED/"inner_tuning_scores.tsv",sep="\t").query("penalty=='ridge'").set_index(KEYS+["C"])
    old_pred=pd.read_csv(MAPPED/"out_of_fold_predictions.tsv",sep="\t",dtype={"PATNO":str}).query("penalty=='ridge'")
    old_features=pd.read_csv(MAPPED/"outer_gene_coefficients.tsv",sep="\t",usecols=KEYS+["penalty","Geneid"]).query("penalty=='ridge'")
    feature_sets={k:g.Geneid.to_numpy() for k,g in old_features.groupby(KEYS)}
    test_sets={k:g.set_index("PATNO") for k,g in old_pred.groupby(KEYS)}
    predictions=[];metrics=[];params=[];tuning=[];checks=[];coefficients=[];inner_saved={}
    for number,split in enumerate(plan,1):
        context={k:split[k] for k in KEYS};key=tuple(context[k] for k in KEYS)
        status("TRAINING",completed_folds=number-1,total_folds=30,elapsed_seconds=time.monotonic()-started,**context)
        print(f"{datetime.now(timezone.utc).isoformat()} {number}/30 {context}",flush=True)
        tr,te=np.array(split["train_indices"]),np.array(split["test_indices"])
        inner=[(np.array(i["train_positions"]),np.array(i["validation_positions"])) for i in split["inner"]]
        best,rows,inner_p=tune(data,tr,inner);tuning.extend(dict(**context,**r) for r in rows)
        for candidate,p in inner_p.items():inner_saved[f"{number}_"+"_".join(map(str,candidate))]=p
        inner_saved[f"{number}_train_indices"]=tr
        for row in rows:
            if row["family"]=="rna":
                reference=old_inner.loc[key+(row["C"],)]
                np.testing.assert_allclose(row["mean_inner_AUROC"],reference.mean_inner_AUROC,atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.fromstring(row["inner_AUROCs"],sep=";"),np.fromstring(reference.inner_AUROCs,sep=";"),atol=1e-12,rtol=0)
        chosen=best["rna"];prior=old_params.loc[key]
        assert chosen["C"]==prior.C
        np.testing.assert_allclose(chosen["threshold"],prior.threshold,atol=1e-8,rtol=1e-8)
        pack=data.pack(tr,te)
        np.testing.assert_array_equal(genes[pack["state"]["indices"]],feature_sets[key])
        previous=test_sets[key].loc[meta.PATNO.iloc[te]]
        np.testing.assert_array_equal(previous.label,y[te])
        for family in FAMILIES:
            chosen=best[family];x,v=matrices(pack,chosen)
            model=fit_model(x,y[tr],chosen["C"],"ridge",0.)
            p=model.predict_proba(v)[:,1]
            if family=="rna":
                np.testing.assert_allclose(p,previous.probability_PD,atol=1e-8,rtol=1e-8)
                checks.append(dict(**context,family=family,max_probability_difference=float(np.max(abs(p-previous.probability_PD))),status="PASS"))
            train_p=model.predict_proba(x)[:,1]
            metrics.append(dict(**context,family=family,n_train=len(tr),n_test=len(te),eligible_genes=0 if family=="blood" else pack["train"].shape[1],
                n_features=x.shape[1],AUROC=float(roc_auc_score(y[te],p)),AP=float(average_precision_score(y[te],p)),
                train_AUROC=float(roc_auc_score(y[tr],train_p)),train_AP=float(average_precision_score(y[tr],train_p))))
            predictions.extend(dict(**context,family=family,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(prob),threshold=chosen["threshold"]) for i,prob in zip(te,p))
            params.append(dict(**context,**chosen,n_features=x.shape[1],optimizer_iterations=int(model.n_iter_[0])))
            state=artifact(model,pack,chosen,genes)
            np.testing.assert_allclose(predict_artifact(state,raw[te],meta.iloc[te]),p,atol=1e-9,rtol=1e-9)
            features=([] if family=="rna" else ["log_wbc","neutrophils_percent","monocytes_percent","eosinophils_percent","basophils_percent"])+([] if family=="blood" else genes[pack["state"]["indices"]].tolist())
            effects=model.coef_[0].copy()
            if family=="rna_blood":effects[:5]*=state["blood_multiplier"]
            coefficients.extend(dict(**context,family=family,feature=f,coefficient_on_standardized_input=float(c)) for f,c in zip(features,effects))
        table("out_of_fold_predictions.tsv",predictions);table("outer_fold_metrics.tsv",metrics)
        table("selected_parameters.tsv",params);table("inner_tuning_scores.tsv",tuning);table("benchmark_reproduction.tsv",checks)
    pred=pd.DataFrame(predictions)
    assert len(pred)==528*3*2*3 and not pred.duplicated(["repeat","protocol","family","PATNO"]).any()
    np.savez_compressed(HERE/"inner_oof_probabilities.npz",**inner_saved)
    table("outer_coefficients.tsv",coefficients)
    status("FINAL_DEVELOPMENT_FITS",completed_folds=30,total_folds=30)
    final_plan=json.loads((MAPPED/"final_tuning_folds.json").read_text())
    final_splits=[(np.array(s["train_indices"]),np.array(s["validation_indices"])) for s in final_plan]
    save("final_tuning_folds.json",final_plan)
    best,rows,_=tune(data,np.arange(528),final_splits);table("final_tuning_scores.tsv",rows)
    pack=data.pack(np.arange(528),np.array([],dtype=int));(HERE/"models").mkdir(exist_ok=True)
    for family in FAMILIES:
        chosen=best[family];x,_=matrices(pack,chosen)
        model=fit_model(x,y,chosen["C"],"ridge",0.)
        state=artifact(model,pack,chosen,genes)
        state.update(training_n=528,candidate_gene_ids=genes[indices],sklearn_version=sklearn.__version__,numpy_version=np.__version__)
        dest=HERE/"models"/f"{family}_ridge.joblib";joblib.dump(state,dest)
        np.testing.assert_allclose(predict_artifact(joblib.load(dest),raw[:5],meta.iloc[:5]),model.predict_proba(x[:5])[:,1],atol=1e-9,rtol=1e-9)
    save("final_parameters.json",list(best.values()))
    status("SUMMARIZING",completed_folds=30,total_folds=30)
    from reporting import report
    result=report(HERE,meta,pred,pd.DataFrame(metrics),device)
    hashes=json.loads((HERE/"input_sha256.json").read_text())
    assert all(sha(p)==v for p,v in hashes.items())
    result.update(status="COMPLETE",participants=528,source_hashes_unchanged=True,validation="PASS",
                  reproduced_RNA_folds=30,measured_blood_predictors=5,
                  elapsed_seconds=time.monotonic()-started,device=str(device),main_model_fits=2373)
    save("summary.json",result)
    save("final_audit.json",dict(status="PASS",RNA_folds_reproduced=30,outer_inference_checks=90,final_artifacts=3,source_hashes_unchanged=True))
    output_files=[p for p in HERE.rglob("*") if p.is_file() and not any(part in ["tmp","cache","matplotlib_cache"] for part in p.parts) and p.name not in ["run.log","status.json","output_sha256.json",".launch.lock"]]
    save("output_sha256.json",{str(p):sha(p) for p in output_files})
    status("COMPLETE",completed_folds=30,total_folds=30,elapsed_seconds=time.monotonic()-started)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):
            device=choose_device()
            if device.type!="cuda":raise RuntimeError("Requested CUDA/RTX is unavailable")
            if "--preflight-only" in sys.argv:
                result=preflight(device);save(f"synthetic_preflight_{device.type}.json",result);print(json.dumps(result,indent=2))
            else:
                if (HERE/"status.json").exists():raise RuntimeError("Training run already exists; refusing overwrite")
                train(device)
    except Exception:
        if "--preflight-only" not in sys.argv and not ((HERE/"status.json").exists() and json.loads((HERE/"status.json").read_text()).get("status")=="COMPLETE"):
            status("FAILED",error=traceback.format_exc())
        traceback.print_exc();sys.exit(1)
