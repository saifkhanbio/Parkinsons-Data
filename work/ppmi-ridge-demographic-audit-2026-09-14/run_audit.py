"""Audit saved ridge predictions without fitting or changing any model."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")

import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

torch.set_num_threads(1)
BOOTSTRAPS = 5000
SEED = 20260914
PROTOCOLS = ["participant_stratified", "batch_grouped"]
GROUPS = ["All", "Female", "Male", "Age <50", "Age 50-69", "Age >=70"]
METRICS = ["AUROC", "AP", "prevalence", "AP_minus_prevalence"]


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame, name):
    frame.to_csv(HERE / name, sep="\t", index=False, float_format="%.12g")


def weighted_metrics(y, scores, weights, device):
    """Rows are bootstrap multiplicities; ties are evaluated at group endpoints."""
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(y) - 1]
    labels = torch.as_tensor(y[order], dtype=torch.float64, device=device)
    endpoints = torch.as_tensor(ends, dtype=torch.long, device=device)
    results = []
    for start in range(0, len(weights), 512):
        w = torch.as_tensor(weights[start:start + 512, order], dtype=torch.float64, device=device)
        tp = torch.cumsum(w * labels, dim=1)[:, endpoints]
        fp = torch.cumsum(w * (1 - labels), dim=1)[:, endpoints]
        dtp = torch.diff(tp, prepend=torch.zeros_like(tp[:, :1]), dim=1)
        dfp = torch.diff(fp, prepend=torch.zeros_like(fp[:, :1]), dim=1)
        positives, negatives = tp[:, -1], fp[:, -1]
        valid = (positives > 0) & (negatives > 0)
        auc = torch.sum(dfp * (tp - 0.5 * dtp), dim=1) / (positives * negatives)
        precision = torch.where(tp + fp > 0, tp / (tp + fp), 0.0)
        ap = torch.sum(dtp * precision, dim=1) / positives
        prevalence = positives / (positives + negatives)
        arr = torch.stack([auc, ap, prevalence, ap - prevalence], dim=1)
        arr[~valid] = float("nan")
        results.append(arr.cpu().numpy())
    return np.concatenate(results)


def validate_metric_engine(device):
    rng = np.random.default_rng(4821)
    errors = []
    for scores in (np.array([0.2, 0.2, 0.9, 0.5, 0.5, 0.1]), np.ones(6), np.arange(6.0)):
        y = np.array([0, 1, 1, 0, 1, 0])
        w = rng.integers(0, 5, size=(32, 6))
        actual = weighted_metrics(y, scores, w, device)
        for j in range(len(w)):
            if np.dot(w[j], y) == 0 or np.dot(w[j], 1-y) == 0:
                assert np.isnan(actual[j]).all()
                continue
            expected = [roc_auc_score(y, scores, sample_weight=w[j]),
                        average_precision_score(y, scores, sample_weight=w[j])]
            errors.extend(abs(actual[j, :2] - expected))
    assert max(errors) < 1e-12
    return float(max(errors))


def bootstrap_weights(meta, protocol, rng):
    n = len(meta)
    if protocol == "participant_stratified":
        weights = np.zeros((BOOTSTRAPS, n), dtype=np.int16)
        for label in (0, 1):
            positions = np.flatnonzero(meta.label.to_numpy() == label)
            weights[:, positions] = rng.multinomial(len(positions), np.full(len(positions), 1/len(positions)), size=BOOTSTRAPS)
        assert np.all(weights.sum(axis=1) == n)
    else:
        levels, codes = np.unique(meta.batch.astype(str), return_inverse=True)
        batch_weights = rng.multinomial(len(levels), np.full(len(levels), 1/len(levels)), size=BOOTSTRAPS)
        weights = batch_weights[:, codes].astype(np.int16)
        for code in range(len(levels)):
            assert np.all(weights[:, codes == code] == batch_weights[:, [code]])
    return weights


def make_figure(summary, counts):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharey=True)
    colors = ["#176B87", "#8B4A78"]
    display = {"All": "All participants", "Female": "Women", "Male": "Men",
               "Age <50": "Age <50", "Age 50-69": "Age 50–69", "Age >=70": "Age ≥70"}
    count_map = counts.set_index("subgroup")
    labels = [f"{display[g]}  ({int(count_map.loc[g, 'n_PD'])}/{int(count_map.loc[g, 'n_Control'])})" for g in GROUPS]
    for col, protocol in enumerate(PROTOCOLS):
        s = summary[summary.protocol == protocol].set_index("subgroup").loc[GROUPS]
        for row, metric in enumerate(["AUROC", "AP"]):
            ax = axes[row, col]
            point = s[f"{metric}_mean"].to_numpy()
            low, high = s[f"{metric}_ci_lower"].to_numpy(), s[f"{metric}_ci_upper"].to_numpy()
            y = np.arange(len(GROUPS))
            ax.hlines(y, low, high, color=colors[col], linewidth=1.8)
            ax.scatter(point, y, s=39, color=colors[col], zorder=3, label="Ridge")
            if metric == "AUROC":
                ax.axvline(.5, color="#8A8A8A", ls="--", lw=1)
            else:
                ax.scatter(s.prevalence, y, marker="|", s=180, color="#707070", label="PD prevalence", zorder=4)
            xmin = max(0, min(.45 if metric == "AUROC" else s.prevalence.min(), np.nanmin(low)) - .035)
            xmax = min(1, np.nanmax(high) + .12)
            ax.set_xlim(xmin, xmax)
            ax.set_yticks(y, labels)
            ax.set_ylim(len(GROUPS)-.5, -.6)
            ax.grid(axis="x", alpha=.17)
            ax.set_xlabel("AUROC" if metric == "AUROC" else "Average precision (AP)")
            for i, value in enumerate(point):
                ax.text(xmax-.008, i, f"{value:.3f}", ha="right", va="center", fontsize=9,
                        bbox={"facecolor":"white", "edgecolor":"none", "pad":1})
            ax.text(-.07, 1.04, "ABCD"[row*2+col], transform=ax.transAxes, fontweight="bold", fontsize=13)
            if row == 0:
                ax.set_title("Participant-stratified folds" if col == 0 else "Batch-grouped folds", pad=15)
            else:
                ax.legend(loc="lower left", bbox_to_anchor=(0,-.34), frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Demographic performance of the existing Hallmark ridge classifier", fontsize=15, y=.975)
    fig.text(.5,.925,"528 participants • Saved held-out predictions • Labels show PD/control counts",ha="center",fontsize=10)
    fig.subplots_adjust(left=.22, right=.975, top=.855, bottom=.19, hspace=.48, wspace=.16)
    fig.text(.22,.048,"Points: mean of three repeat-specific pooled out-of-fold metrics; lines: conditional 95% bootstrap intervals.\nAge at RNA collection. No models were refitted. AP reference marks show subgroup PD prevalence.",fontsize=9,linespacing=1.6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"demographic_performance.{ext}", dpi=600, facecolor="white")
    fig.savefig(HERE / "demographic_performance_preview.png", dpi=125, facecolor="white")
    plt.close(fig)


def main():
    if (HERE / "status.json").exists():
        existing = json.loads((HERE / "status.json").read_text())
        if existing.get("state") == "COMPLETE":
            raise RuntimeError("Completed audit exists; use a new folder for a different analysis.")
    save_json("status.json", {"state":"RUNNING", "started":now(), "stage":"validate_inputs"})
    mapped = ROOT / "work/ppmi-mapped-gene-classifier-2026-09-12"
    sources = [mapped/name for name in ("out_of_fold_predictions.tsv", "outer_fold_metrics.tsv", "primary_summary.tsv", "fold_plan.json")]
    sources.append(ROOT / "work/ppmi-classifier-2026-09-12/metadata.tsv")
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in sources}
    save_json("provenance.json", {"created":now(), "source_sha256":hashes,
              "script_sha256":sha256(Path(__file__)), "plan_sha256":sha256(HERE / "README.md"),
              "seed":SEED, "bootstrap_draws":BOOTSTRAPS})
    meta = pd.read_csv(sources[-1], sep="\t", dtype={"PATNO":str})
    pred = pd.read_csv(sources[0], sep="\t", dtype={"PATNO":str})
    pred = pred[pred.penalty == "ridge"].copy()
    assert len(meta) == 528 and meta.PATNO.is_unique
    assert meta.label.value_counts().to_dict() == {1:358, 0:170}
    assert meta.sex.value_counts().to_dict() == {"Male":344, "Female":184}
    assert np.array_equal(meta.label.to_numpy(), (meta.group == "PD").astype(int).to_numpy())
    assert np.isfinite(meta.age_collection_years).all() and meta.batch.notna().all()
    assert len(pred) == 528*3*2 and not pred.duplicated(["protocol","repeat","PATNO"]).any()
    assert set(pred.protocol) == set(PROTOCOLS) and set(pred.repeat) == {1,2,3}
    assert np.isfinite(pred.probability_PD).all() and pred.probability_PD.between(0,1).all()
    labels = meta.set_index("PATNO").label
    assert (pred.PATNO.map(labels).to_numpy() == pred.label.to_numpy()).all()
    for _, block in pred.groupby(["protocol", "repeat"]):
        assert set(block.PATNO) == set(meta.PATNO)
    plans = json.loads(sources[3].read_text())
    assert len(plans) == 30
    for plan in plans:
        train, test = np.array(plan["train_indices"]), np.array(plan["test_indices"])
        assert not set(train) & set(test) and set(train) | set(test) == set(range(528))
        block = pred[(pred.protocol==plan["protocol"]) & (pred.repeat==plan["repeat"]) & (pred.fold==plan["fold"])]
        assert set(block.PATNO) == set(meta.iloc[test].PATNO)
        if plan["protocol"] == "batch_grouped":
            assert not set(meta.iloc[train].batch) & set(meta.iloc[test].batch)
    age = meta.age_collection_years.to_numpy()
    masks = {"All":np.ones(528,dtype=bool), "Female":(meta.sex=="Female").to_numpy(),
             "Male":(meta.sex=="Male").to_numpy(), "Age <50":age<50,
             "Age 50-69":(age>=50)&(age<70), "Age >=70":age>=70}
    assert np.all(masks["Age <50"].astype(int)+masks["Age 50-69"]+masks["Age >=70"]==1)
    counts = []
    for group, mask in masks.items():
        x=meta[mask]; cases=int(x.label.sum()); controls=len(x)-cases
        counts.append({"subgroup":group,"n":len(x),"n_PD":cases,"n_Control":controls,
                       "prevalence":cases/len(x),"age_min":x.age_collection_years.min(),
                       "age_max":x.age_collection_years.max(),"n_batches":x.batch.nunique(),
                       "sparse":min(cases,controls)<20})
    counts=pd.DataFrame(counts); table(counts,"subgroup_counts.tsv")
    membership=meta[["PATNO","group","label","sex","age_collection_years","batch"]].copy()
    membership["age_band"]=np.select([age<50,age<70],["Age <50","Age 50-69"],default="Age >=70")
    table(membership,"participant_subgroups.tsv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    synthetic_error=validate_metric_engine(device)
    print(f"Input/fold checks passed; {device=}; bootstrap draws={BOOTSTRAPS}",flush=True)
    fold_rows=[]
    for (protocol, repeat, fold), block in pred.groupby(["protocol","repeat","fold"]):
        positions=meta.set_index("PATNO").index.get_indexer(block.PATNO)
        for group, mask in masks.items():
            sub=block[mask[positions]]
            good=sub.label.nunique()==2
            fold_rows.append({"protocol":protocol,"repeat":repeat,"fold":fold,"subgroup":group,
                "n":len(sub),"n_PD":int(sub.label.sum()),"n_Control":len(sub)-int(sub.label.sum()),
                "AUROC":roc_auc_score(sub.label,sub.probability_PD) if good else np.nan,
                "AP":average_precision_score(sub.label,sub.probability_PD) if good else np.nan,
                "valid":good})
    folds=pd.DataFrame(fold_rows);table(folds,"metrics_by_fold.tsv")
    original=pd.read_csv(sources[1],sep="\t").query("penalty=='ridge'")
    check=folds[folds.subgroup=="All"].merge(original,on=["protocol","repeat","fold"],validate="one_to_one")
    fold_error=float(np.max(abs(check.AUROC-check.test_AUROC)))
    assert fold_error<1e-12
    base=pd.read_csv(sources[2],sep="\t").query("penalty=='ridge'").set_index("protocol")
    benchmark=[]
    for protocol in PROTOCOLS:
        q=folds[(folds.protocol==protocol)&(folds.subgroup=="All")]
        auc=float(q.groupby("repeat").AUROC.mean().mean())
        assert abs(auc-base.loc[protocol,"AUROC_mean"])<1e-12
        benchmark.append({"protocol":protocol,"original_AUROC":base.loc[protocol,"AUROC_mean"],
                          "reproduced_mean_within_fold_AUROC":auc,"mean_within_fold_AP":q.groupby("repeat").AP.mean().mean()})
    table(pd.DataFrame(benchmark),"benchmark_reproduction.tsv")
    save_json("status.json",{"state":"RUNNING","updated":now(),"stage":"bootstrap"})
    summaries=[]; repeats=[]; contrasts=[]; boot_archive={}; real_errors=[]
    for pi,protocol in enumerate(PROTOCOLS):
        rng=np.random.default_rng(SEED+1000*pi)
        weights=bootstrap_weights(meta,protocol,rng)
        boot_by_group={}; point_by_group={}
        for group, mask in masks.items():
            y=meta.label.to_numpy()[mask]; w=weights[:,mask]
            bs=[]; points=[]
            for repeat in (1,2,3):
                block=pred[(pred.protocol==protocol)&(pred.repeat==repeat)].set_index("PATNO").loc[meta.PATNO]
                scores=block.probability_PD.to_numpy()[mask]
                all_metrics=weighted_metrics(y,scores,np.vstack([np.ones((1,len(y))),w]),device)
                point, draws=all_metrics[0],all_metrics[1:]
                for j in [0,1,2,17,53]:
                    if np.isfinite(draws[j,:2]).all():
                        ref=[roc_auc_score(y,scores,sample_weight=w[j]),average_precision_score(y,scores,sample_weight=w[j])]
                        real_errors.extend(abs(draws[j,:2]-ref))
                assert np.allclose(point[:2],[roc_auc_score(y,scores),average_precision_score(y,scores)],atol=1e-12,rtol=0)
                points.append(point);bs.append(draws)
                repeats.append(dict(protocol=protocol,subgroup=group,repeat=repeat,**dict(zip(METRICS,point))))
            point=np.mean(points,axis=0);draws=np.mean(bs,axis=0)
            point_by_group[group]=point;boot_by_group[group]=draws
            boot_archive[f"{protocol}__{group}"]=draws
            row=dict(protocol=protocol,**counts.set_index("subgroup").loc[group].to_dict(),subgroup=group)
            valid=np.isfinite(draws).all(axis=1)
            row["bootstrap_valid"]=int(valid.sum());row["bootstrap_invalid"]=int((~valid).sum())
            for mi,metric in enumerate(METRICS):
                row[f"{metric}_mean"]=point[mi]
                row[f"{metric}_ci_lower"],row[f"{metric}_ci_upper"]=np.quantile(draws[valid,mi],[.025,.975])
            subfolds=folds[(folds.protocol==protocol)&(folds.subgroup==group)]
            row["valid_outer_folds"]=int(subfolds.valid.sum())
            row["AUROC_mean_within_fold"]=subfolds.groupby("repeat").AUROC.mean().mean()
            row["AP_mean_within_fold"]=subfolds.groupby("repeat").AP.mean().mean()
            summaries.append(row)
        for group,reference in [("Female","Male"),("Age <50","Age 50-69"),("Age >=70","Age 50-69")]:
            diff=boot_by_group[group]-boot_by_group[reference]
            good=np.isfinite(diff).all(axis=1)
            for mi,metric in enumerate(METRICS):
                if metric=="prevalence":continue
                low,high=np.quantile(diff[good,mi],[.025,.975])
                contrasts.append({"protocol":protocol,"comparison":f"{group} minus {reference}","metric":metric,
                    "difference":point_by_group[group][mi]-point_by_group[reference][mi],
                    "ci_lower":low,"ci_upper":high,"bootstrap_valid":int(good.sum()),
                    "interval_excludes_zero":bool(low>0 or high<0)})
        print(f"Completed {protocol} subgroup bootstrap",flush=True)
    assert max(real_errors)<1e-12
    summary=pd.DataFrame(summaries);contrast=pd.DataFrame(contrasts)
    table(summary,"subgroup_performance.tsv");table(pd.DataFrame(repeats),"metrics_by_repeat.tsv")
    table(contrast,"subgroup_contrasts.tsv")
    np.savez_compressed(HERE/"bootstrap_metrics.npz",metric_names=np.array(METRICS),**boot_archive)
    make_figure(summary,counts)
    lines=["# Hallmark ridge demographic audit", "", "Completed "+now()+".", "",
        "No models were fitted. All 528 original participants and saved ridge predictions were retained.", "",
        "## Overall benchmark reproduction", "",
        "The original mean-within-outer-fold AUROCs reproduce to numerical precision. AP below uses the same fold averaging.", "",
        "| Protocol | Original/reproduced AUROC | Mean within-fold AP |", "| --- | ---: | ---: |"]
    for row in benchmark:
        lines.append(f"| {row['protocol']} | {row['original_AUROC']:.3f} | {row['mean_within_fold_AP']:.3f} |")
    lines += ["", "## Subgroup performance", "",
        "This table averages three repeat-specific pooled out-of-fold metrics. Its overall row is a different estimand from the original fold-averaged benchmark above. Intervals are conditional 95% bootstrap intervals.", "",
        "| Protocol | Group | PD / controls | AUROC (95% interval) | AP (95% interval) | PD prevalence |", "| --- | --- | ---: | --- | --- | ---: |"]
    for _,row in summary.iterrows():
        lines.append(f"| {row.protocol} | {row.subgroup} | {row.n_PD} / {row.n_Control} | {row.AUROC_mean:.3f} ({row.AUROC_ci_lower:.3f}–{row.AUROC_ci_upper:.3f}) | {row.AP_mean:.3f} ({row.AP_ci_lower:.3f}–{row.AP_ci_upper:.3f}) | {row.prevalence:.3f} |")
    lines += ["", "## Exploratory contrasts", "",
        "Contrasts use the same bootstrap draw across repeats and subgroups. They do not establish that a separate subgroup model would outperform the pooled model. AP differences also depend on disease prevalence.", "",
        "| Protocol | Comparison | Metric | Difference (95% interval) |", "| --- | --- | --- | --- |"]
    for _,row in contrast.iterrows():
        lines.append(f"| {row.protocol} | {row.comparison} | {row.metric} | {row.difference:+.3f} ({row.ci_lower:+.3f}–{row.ci_upper:+.3f}) |")
    lines += ["", "## Interpretation boundaries", "",
        "The age groups were fixed before subgroup scoring and use age at RNA collection. Each participant retains all three predictions during resampling. Batch-grouped uncertainty resamples whole phase-plate batches. Bootstrap intervals condition on the saved predictions and omit retraining variability. No formal multiplicity-adjusted subgroup hypothesis testing was performed. Subgroup differences, including intervals excluding zero, are exploratory.", "",
        "AP denotes average precision, not trapezoidal PR-AUC. The PD prevalence reference varies across subgroups. AP minus prevalence is provided as descriptive context and is not a universal prevalence-adjusted discrimination measure.", "",
        "The cohort has been examined in earlier model comparisons. This audit provides no evidence that a newly fitted demographic model improves performance. Later experiments require matched nested comparisons; none was launched here.", "",
        "![Demographic performance](demographic_performance.png)", "",
        "## Validation", "",
        f"All 30 original ridge outer-fold AUROCs reproduced (maximum absolute error {fold_error:.3g}). Participant/fold/batch alignment and count assertions passed. Synthetic weighted/tied-score checks and real bootstrap checks matched sklearn within 1e-12. Execution used {device} with one numerical CPU thread.", "",
        "See `validation.json`, `provenance.json`, and the README for methods and source paths.", ""]
    (HERE/"RESULTS.md").write_text("\n".join(lines))
    after={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    assert after==hashes
    validate={"passed":True,"completed":now(),"n_participants":528,"n_ridge_predictions":len(pred),
        "n_outer_folds":30,"source_hashes_unchanged":True,"max_original_fold_AUROC_error":fold_error,
        "max_synthetic_metric_error":synthetic_error,"max_real_bootstrap_metric_error":float(max(real_errors)),
        "bootstrap_draws":BOOTSTRAPS,"minimum_valid_bootstraps":int(summary.bootstrap_valid.min()),
        "device":device,"cuda_available":torch.cuda.is_available(),"model_refits":0,
        "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
        "sklearn":sklearn.__version__,"torch":torch.__version__,"subgroup_rows":len(summary)}
    save_json("validation.json",validate)
    save_json("status.json",{"state":"COMPLETE","completed":now(),"model_refits":0})
    print(summary[["protocol","subgroup","n_PD","n_Control","AUROC_mean","AP_mean","prevalence"]].to_string(index=False),flush=True)
    print("COMPLETE: source hashes unchanged; all validation checks passed.",flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        if not (HERE/"status.json").exists() or json.loads((HERE/"status.json").read_text()).get("state")!="COMPLETE":
            save_json("status.json",{"state":"FAILED","updated":now(),"traceback":traceback.format_exc()})
        traceback.print_exc()
        sys.exit(1)
