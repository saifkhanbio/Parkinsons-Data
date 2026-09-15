"""Paired fixed-prediction uncertainty and reports for Step 2."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from demographic_models import FAMILIES,evaluate

PROTOCOLS=["participant_stratified","batch_grouped"]
BOOTSTRAPS=5000
SEED=20260914


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


def bootstrap_weights(meta,protocol,rng):
    if protocol=="participant_stratified":
        weights=np.zeros((BOOTSTRAPS,len(meta)),dtype=np.int16)
        for label in [0,1]:
            where=np.flatnonzero(meta.label.to_numpy()==label)
            weights[:,where]=rng.multinomial(len(where),np.full(len(where),1/len(where)),size=BOOTSTRAPS)
        assert np.all(weights.sum(1)==len(meta))
    else:
        levels,codes=np.unique(meta.batch.astype(str),return_inverse=True)
        w=rng.multinomial(len(levels),np.full(len(levels),1/len(levels)),size=BOOTSTRAPS)
        weights=w[:,codes].astype(np.int16)
        for i in range(len(levels)):assert np.all(weights[:,codes==i]==w[:,[i]])
    return np.vstack([np.ones((1,len(meta))),weights])


def add_interval(row,values):
    valid=np.isfinite(values[1:]).all(axis=1)
    row.update(bootstrap_valid=int(valid.sum()),bootstrap_invalid=int((~valid).sum()))
    assert valid.sum()>BOOTSTRAPS*.8,"Too many undefined bootstrap draws"
    for j,metric in enumerate(["AUROC","AP"]):
        row[f"{metric}_mean"]=float(values[0,j])
        low,high=np.quantile(values[1:][valid,j],[.025,.975])
        row[f"{metric}_ci_lower"]=float(low);row[f"{metric}_ci_upper"]=float(high)
    return row


def report(root,meta,pred,fold_metrics,device):
    def table(name,x):pd.DataFrame(x).to_csv(root/name,sep="\t",index=False)
    y=meta.label.to_numpy();age=meta.age_collection_years.to_numpy()
    masks={"All":np.ones(len(meta),dtype=bool),"Female":meta.sex.eq("Female").to_numpy(),
           "Male":meta.sex.eq("Male").to_numpy(),"Age <50":age<50,
           "Age 50-69":(age>=50)&(age<70),"Age >=70":age>=70}
    counts=[]
    for group,mask in masks.items():
        cases=int(y[mask].sum());n=int(mask.sum())
        counts.append(dict(subgroup=group,n=n,n_PD=cases,n_Control=n-cases,prevalence=cases/n))
    counts=pd.DataFrame(counts);table("subgroup_counts.tsv",counts)
    rng=np.random.default_rng(553)
    test_y=np.array([0,1,0,1,0,1]);test_scores=np.array([.1,.1,.4,.4,.9,.9])
    w=rng.integers(1,5,size=(30,6));calc=weighted_metrics(test_y,test_scores,w,device)
    reference=np.array([[roc_auc_score(test_y,test_scores,sample_weight=q),average_precision_score(test_y,test_scores,sample_weight=q)] for q in w])
    metric_error=float(np.max(abs(calc-reference)));assert metric_error<1e-12
    repeats=fold_metrics.groupby(["protocol","family","repeat"],as_index=False)[["AUROC","AP","train_AUROC","train_AP"]].mean()
    table("primary_per_repeat.tsv",repeats)
    primary=[];subgroups=[];contrasts=[];subgroup_delta=[];secondary=[];archive={};real_error=[]
    for pi,protocol in enumerate(PROTOCOLS):
        weights=bootstrap_weights(meta,protocol,np.random.default_rng(SEED+1000*pi))
        model_primary={};model_subgroups={}
        for family in FAMILIES:
            draws=[];pooled={g:[] for g in masks}
            for repeat in [1,2,3]:
                part=pred[(pred.protocol==protocol)&(pred.family==family)&(pred.repeat==repeat)].set_index("PATNO").loc[meta.PATNO]
                assert len(part)==528 and np.array_equal(part.label,y)
                scores=part.probability_PD.to_numpy();assignment=part.fold.to_numpy()
                fold_draws=[]
                for fold in [1,2,3,4,5]:
                    mask=assignment==fold
                    values=weighted_metrics(y[mask],scores[mask],weights[:,mask],device)
                    actual=fold_metrics[(fold_metrics.protocol==protocol)&(fold_metrics.family==family)&(fold_metrics.repeat==repeat)&(fold_metrics.fold==fold)].iloc[0]
                    np.testing.assert_allclose(values[0],[actual.AUROC,actual.AP],atol=1e-12,rtol=0)
                    fold_draws.append(values)
                draws.append(np.mean(fold_draws,axis=0))
                for group,mask in masks.items():
                    values=weighted_metrics(y[mask],scores[mask],weights[:,mask],device)
                    pooled[group].append(values)
                    for sample in [1,17]:
                        if np.isfinite(values[sample]).all():
                            ww=weights[sample,mask]
                            expected=[roc_auc_score(y[mask],scores[mask],sample_weight=ww),average_precision_score(y[mask],scores[mask],sample_weight=ww)]
                            real_error.extend(abs(values[sample]-expected))
                for mode,t in [("fixed_0_5",.5),("inner_selected",part.threshold.to_numpy())]:
                    secondary.append(dict(protocol=protocol,family=family,repeat=repeat,threshold_mode=mode,**evaluate(y,scores,t)))
            mean=np.mean(draws,axis=0);model_primary[family]=mean
            archive[f"primary__{protocol}__{family}"]=mean
            row=add_interval(dict(protocol=protocol,family=family),mean)
            rep=repeats[(repeats.protocol==protocol)&(repeats.family==family)]
            for metric in ["AUROC","AP"]:
                row[f"{metric}_repeat_min"]=float(rep[metric].min());row[f"{metric}_repeat_max"]=float(rep[metric].max())
            row.update(train_AUROC=float(rep.train_AUROC.mean()),train_AP=float(rep.train_AP.mean()))
            primary.append(row)
            for group in masks:
                val=np.mean(pooled[group],axis=0);model_subgroups[(family,group)]=val
                archive[f"pooled__{protocol}__{family}__{group}"]=val
                subgroups.append(add_interval(dict(protocol=protocol,family=family,subgroup=group,**counts.set_index("subgroup").loc[group].to_dict()),val))
        for family,reference_family in [("rna_demographic","rna"),("rna_demographic","demographic"),("demographic","rna")]:
            contrasts.append(add_interval(dict(protocol=protocol,comparison=f"{family} minus {reference_family}"),model_primary[family]-model_primary[reference_family]))
        for group in masks:
            subgroup_delta.append(add_interval(dict(protocol=protocol,subgroup=group,comparison="rna_demographic minus rna"),model_subgroups[("rna_demographic",group)]-model_subgroups[("rna",group)]))
        print(f"Paired bootstrap complete: {protocol}",flush=True)
    assert max(real_error)<1e-12
    primary=pd.DataFrame(primary);contrast=pd.DataFrame(contrasts)
    table("primary_summary.tsv",primary);table("paired_primary_comparisons.tsv",contrast)
    table("subgroup_performance.tsv",subgroups);table("subgroup_paired_comparisons.tsv",subgroup_delta)
    table("secondary_pooled_OOF_metrics.tsv",secondary)
    np.savez_compressed(root/"bootstrap_metrics.npz",metric_names=np.array(["AUROC","AP"]),**archive)
    paired=[]
    for family,ref in [("rna_demographic","rna"),("rna_demographic","demographic")]:
        a=fold_metrics[fold_metrics.family==family].set_index(["protocol","repeat","fold"])
        b=fold_metrics[fold_metrics.family==ref].set_index(["protocol","repeat","fold"])
        diff=(a[["AUROC","AP"]]-b[["AUROC","AP"]]).reset_index();diff["comparison"]=f"{family} minus {ref}"
        paired.append(diff)
    pf=pd.concat(paired);table("paired_fold_differences.tsv",pf)
    table("paired_repeat_differences.tsv",pf.groupby(["protocol","comparison","repeat"],as_index=False)[["AUROC","AP"]].mean())
    make_figure(root,primary,repeats)
    labels={"rna":"RNA-only ridge","demographic":"Age/sex-only ridge","rna_demographic":"RNA + age/sex ridge"}
    lines=["# Step 2: age and sex added to Hallmark ridge", "",
        "Same 528 participants and all original nested folds. All preprocessing, gene eligibility, regularization grid and class weighting remain fixed; age and sex are the only added predictors.","",
        "## Primary comparison","",
        "AUROC and AP are mean within-outer-fold metrics, averaged across three repeats. Parentheses show conditional 95% bootstrap intervals, not repeat ranges. AP is average precision.","",
        "| Protocol | Model | AUROC (95% interval) | AP (95% interval) |","| --- | --- | --- | --- |"]
    for r in primary.itertuples(index=False):
        lines.append(f"| {r.protocol} | {labels[r.family]} | {r.AUROC_mean:.3f} ({r.AUROC_ci_lower:.3f}–{r.AUROC_ci_upper:.3f}) | {r.AP_mean:.3f} ({r.AP_ci_lower:.3f}–{r.AP_ci_upper:.3f}) |")
    lines += ["","## Paired changes","","| Protocol | Comparison | AUROC change (95% interval) | AP change (95% interval) |","| --- | --- | --- | --- |"]
    for r in contrast.itertuples(index=False):
        lines.append(f"| {r.protocol} | {r.comparison} | {r.AUROC_mean:+.3f} ({r.AUROC_ci_lower:+.3f}–{r.AUROC_ci_upper:+.3f}) | {r.AP_mean:+.3f} ({r.AP_ci_lower:+.3f}–{r.AP_ci_upper:+.3f}) |")
    lines += ["","## Demographic reporting","",
        "Subgroup tables use mean repeat-specific pooled out-of-fold metrics, as in Step 1. This differs from the primary within-fold estimand above. Counts and prevalence are in subgroup_counts.tsv; matched RNA+demographic-minus-RNA changes and conditional intervals are in subgroup_paired_comparisons.tsv.","",
        "## Validation and interpretation","",
        "The original RNA-only inner AUROCs, selected regularization, gene identities, thresholds and predictions reproduce in all 30 folds. The initial age/sex control reproduces in its ten repeat-1 folds. All model fits converged; all artifacts passed inference roundtrips. Preflight excludes held-out influence on filtering, scaling, tuning and thresholds.","",
        f"Bootstrap uncertainty uses {BOOTSTRAPS:,} paired draws from fixed predictions, resampling participants within diagnosis or whole phase-plate batches. The same multiplicities apply to every model and repeat. Minimum valid primary draws: {int(primary.bootstrap_valid.min()):,}; unavailable draws are counted in the tables. These descriptive intervals omit model-retraining and adaptive model-selection variability.","",
        "No class weighting, demographic interaction, separate-sex model, random forest or DE screening is included. The maximum observed score across sequential analyses is not an unbiased estimate of an algorithm-selection procedure. No change in clinical utility or disability outcomes is established.","",
        "![Model comparison](demographic_model_comparison.png)","",
        "Three final full-cohort development artifacts and a metadata-aware inference script are saved in this directory; their fits do not add performance evidence.",""]
    (root/"RESULTS.md").write_text("\n".join(lines))
    validation=dict(weighted_metric_max_error=metric_error,real_bootstrap_max_error=float(max(real_error)),
                    bootstrap_draws=BOOTSTRAPS,minimum_valid_primary_draws=int(primary.bootstrap_valid.min()),
                    model_boundaries="PASS",primary_rows=len(primary),subgroup_rows=len(subgroups))
    (root/"report_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    return dict(primary_results=primary.to_dict(orient="records"),paired_results=contrast.to_dict(orient="records"),report_validation=validation)


def make_figure(root,primary,repeats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.0))
    colors=["#247BA0","#777777","#1B8A6B"]
    for col,protocol in enumerate(PROTOCOLS):
        s=primary[primary.protocol==protocol].set_index("family").loc[FAMILIES]
        for row,metric in enumerate(["AUROC","AP"]):
            ax=axes[row,col]
            for i,family in enumerate(FAMILIES):
                low,high=s.loc[family,f"{metric}_ci_lower"],s.loc[family,f"{metric}_ci_upper"]
                point=s.loc[family,f"{metric}_mean"]
                ax.vlines(i,low,high,color=colors[i],lw=2)
                ax.scatter(i,point,color=colors[i],s=55,zorder=3)
                rr=repeats[(repeats.protocol==protocol)&(repeats.family==family)].sort_values("repeat")[metric]
                ax.scatter(i+np.array([-.08,0,.08]),rr,s=16,color=colors[i],alpha=.45,zorder=2)
                ax.annotate(f"{point:.3f}",(i,high),xytext=(0,7),textcoords="offset points",ha="center",fontsize=10)
            if metric=="AUROC":ax.axhline(.5,color="#AAAAAA",ls="--",lw=1)
            else:ax.axhline(358/528,color="#AAAAAA",ls="--",lw=1)
            ax.set_xticks(range(3),["RNA only","Age + sex","RNA +\nage/sex"])
            ax.set_xlim(-.4,2.4)
            low=min(.5 if metric=="AUROC" else 358/528,s[f"{metric}_ci_lower"].min())-.04
            high=min(1,s[f"{metric}_ci_upper"].max()+.075)
            ax.set_ylim(max(0,low),high)
            ax.grid(axis="y",alpha=.15)
            ax.set_ylabel("Mean within-fold "+("AUROC" if metric=="AUROC" else "average precision"))
            ax.text(-.11,1.05,"ABCD"[row*2+col],transform=ax.transAxes,fontweight="bold",fontsize=13)
            if row==0:ax.set_title("Participant-stratified folds" if col==0 else "Batch-grouped folds",pad=16)
    fig.suptitle("Adding age and sex to the Hallmark ridge classifier",fontsize=15,y=.97)
    fig.text(.5,.92,"528 participants • Same genes and nested folds • Three repeats",ha="center",fontsize=10)
    fig.subplots_adjust(left=.10,right=.975,top=.83,bottom=.17,hspace=.43,wspace=.28)
    fig.text(.10,.048,"Large points: mean over repeats. Small points: repeat estimates. Bars: conditional 95% bootstrap intervals.\nDashed lines: AUROC 0.5 or cohort PD prevalence (AP panel). AP denotes average precision.",fontsize=9,linespacing=1.6)
    for ext in ["png","pdf","svg"]:fig.savefig(root/f"demographic_model_comparison.{ext}",dpi=600,facecolor="white")
    fig.savefig(root/"demographic_model_comparison_preview.png",dpi=120,facecolor="white")
    plt.close(fig)
