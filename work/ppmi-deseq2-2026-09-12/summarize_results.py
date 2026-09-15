"""Validate saved results, compare sensitivity models, and produce a report."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["primary", "medication_timing", "phase", "usable_bases", "all_579"]


def markdown_table(frame):
    def value(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v).replace("|", "/")
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(value(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main():
    summaries = json.loads((OUT / "model_summaries.json").read_text())
    enrichment = json.loads((OUT / "enrichment_summary.json").read_text())
    source = json.loads((OUT / "gene_set_provenance.json").read_text())
    influence = json.loads((OUT / "influence_summary.json").read_text())
    gpu = json.loads((OUT / "gpu_validation.json").read_text())
    assert gpu["status"] == "PASSED" and gpu["exact_agreement_with_CPU_count_checks"]
    deadline = time.monotonic() + 300
    while not (OUT / "gpu_result_validation.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("CUDA result validation is missing; run gpu_postprocess.py before reporting")
        time.sleep(2)
    gpu_results = json.loads((OUT / "gpu_result_validation.json").read_text())
    assert gpu_results["status"] == "PASSED"
    gpu_comparison = pd.read_csv(OUT / "gpu_sensitivity_summary.tsv", sep="\t").set_index("model")
    mapping = pd.read_csv(OUT / "msigdb_gene_mapping.tsv", sep="\t", dtype=str)
    symbols = mapping.dropna(subset=["ensembl_gene", "gene_symbol"]).groupby("ensembl_gene").gene_symbol.agg(
        lambda x: ";".join(sorted(set(x))))
    tables = {}
    for name in MODELS:
        table = pd.read_csv(OUT / name / "results.tsv", sep="\t")
        assert table.Geneid.is_unique
        assert table.loc[~table.beta_converged, ["pvalue", "padj", "stat"]].isna().all().all()
        assert len(table) == 21888
        assert table.significant_FDR05.equals(table.padj.lt(0.05))
        assert int(table.significant_FDR05.sum()) == summaries[name]["significant_FDR05"]
        assert table.pvalue.dropna().between(0, 1).all()
        assert table.padj.dropna().between(0, 1).all()
        assert name in gpu_results["model_BH_max_absolute_errors"]
        assert np.isfinite(table.loc[table.pvalue.notna(), ["log2FoldChange", "lfcSE", "stat"]]).all().all()
        assert (table.loc[table.pvalue.notna(), "lfcSE"] > 0).all()
        table.insert(2, "gene_symbol", table.ensembl_id.map(symbols))
        gene_influence = pd.read_csv(OUT / name / "gene_influence.tsv", sep="\t")
        table = table.merge(gene_influence, on="Geneid", how="left", validate="one_to_one")
        table.to_csv(OUT / name / "results_annotated.tsv", sep="\t", index=False)
        tables[name] = table.set_index("Geneid")
    primary = tables["primary"]
    common_significant = set(primary.index[primary.significant_FDR05])
    comparison = []
    wide = primary[["ensembl_id", "gene_symbol", "log2FoldChange", "padj"]].rename(
        columns={"log2FoldChange": "primary_log2FC", "padj": "primary_padj"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for name, ax in zip(MODELS[1:], axes.flat):
        current = tables[name].reindex(primary.index)
        assert set(current.index) == set(primary.index)
        valid = primary.pvalue.notna() & current.pvalue.notna()
        significant = primary.significant_FDR05 & current.significant_FDR05
        same = np.sign(primary.log2FoldChange) == np.sign(current.log2FoldChange)
        primary_sig_comparable = primary.significant_FDR05 & valid
        row = dict(model=name, **gpu_comparison.loc[name].to_dict())
        comparison.append(row)
        common_significant &= set(current.index[current.significant_FDR05 & same])
        wide[name + "_log2FC"] = current.log2FoldChange
        wide[name + "_padj"] = current.padj
        ax.scatter(primary.loc[valid, "log2FoldChange"], current.loc[valid, "log2FoldChange"],
                   s=3, alpha=0.25, color="#287F9C", rasterized=True)
        limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(limits, limits, color="grey", linewidth=1)
        ax.set(xlabel="Primary log2 fold change", ylabel=f"{name}: log2 fold change",
               title=f"{name}: r = {row['log2FC_pearson']:.3f}")
    fig.savefig(OUT / "sensitivity_effects.png", dpi=180)
    fig.savefig(OUT / "sensitivity_effects.pdf")
    plt.close(fig)
    wide["significant_same_direction_all_models"] = wide.index.isin(common_significant)
    assert len(common_significant) == gpu_results["robust_gene_count"]
    wide.to_csv(OUT / "gene_sensitivity_comparison.tsv", sep="\t")
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(OUT / "sensitivity_summary.tsv", sep="\t", index=False)

    hallmark = pd.read_csv(OUT / "primary/hallmark_enrichment.tsv", sep="\t")
    gobp = pd.read_csv(OUT / "primary/gobp_enrichment.tsv", sep="\t")
    hw = hallmark.set_index("pathway")[["NES", "padj"]].add_prefix("primary_")
    for name in MODELS[1:]:
        h = pd.read_csv(OUT / name / "hallmark_enrichment.tsv", sep="\t").set_index("pathway")
        hw = hw.join(h[["NES", "padj"]].add_prefix(name + "_"), how="outer")
    hw["significant_same_direction_all_models"] = np.logical_and.reduce(
        [hw[name + "_padj"].lt(0.05) & (np.sign(hw[name + "_NES"]) == np.sign(hw.primary_NES)) for name in MODELS])
    hw.to_csv(OUT / "hallmark_sensitivity_comparison.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for data, ax, title, prefix in [(hallmark, axes[0], "Hallmark", "HALLMARK_"), (gobp, axes[1], "GO biological process", "GOBP_")]:
        top = data.sort_values("padj").head(12).sort_values("NES")
        labels = top.pathway.str.replace(prefix, "", regex=False).str.replace("_", " ").str.lower()
        # Wrap long GO names to keep the exported figure legible.
        import textwrap
        labels = ["\n".join(textwrap.wrap(label, 38)) for label in labels]
        ax.barh(labels, top.NES, color=np.where(top.NES > 0, "#D86738", "#287F9C"))
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set(title=title + ": 12 smallest adjusted p values", xlabel="Normalized enrichment score (positive: PD)")
        ax.tick_params(axis="y", labelsize=8)
    fig.savefig(OUT / "enrichment_overview.png", dpi=180)
    fig.savefig(OUT / "enrichment_overview.pdf")
    plt.close(fig)

    p = summaries["primary"]
    model_table = pd.DataFrame([dict(Model=name, N=summaries[name]["n"],
                                    Higher_PD=summaries[name]["higher_in_PD"],
                                    Lower_PD=summaries[name]["lower_in_PD"],
                                    FDR05=summaries[name]["significant_FDR05"]) for name in MODELS])
    top_genes = primary.reset_index().head(15)[["Geneid", "gene_symbol", "log2FoldChange", "lfcSE", "padj"]].fillna("")
    prefilter = pd.read_csv(OUT / "gene_prefilter.tsv", sep="\t")
    assert int(prefilter.prefilter_pass.sum()) == len(primary)
    hashes = json.loads((OUT / "input_sha256.json").read_text())
    for relative, expected in hashes.items():
        digest = hashlib.sha256((OUT.parent.parent / relative).read_bytes()).hexdigest()
        assert digest == expected, relative
    validation = dict(status="COMPLETE", models=5, genes_prefiltered=len(primary),
                      nonconverged_genes={name: summaries[name]["final_nonconverged"] for name in MODELS},
                      all_reported_tests_converged=True, input_hashes_unchanged=True,
                      GPU_validation=gpu["device"],
                      GPU_result_postprocessing=gpu_results["operations"],
                      genes_significant_same_direction_all_models=len(common_significant),
                      hallmark_significant_same_direction_all_models=int(hw.significant_same_direction_all_models.sum()))
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    report = '# Local analysis report\n\nConsult the locally generated tables and diagnostics.\n'
    if (OUT / "c2_c5bp/README.md").exists():
        report += "\n## Additional requested collections\n\nSee [C2 and C5 biological-process enrichment](c2_c5bp/README.md) for the primary-model extension.\n"
    (OUT / "README.md").write_text(report)
    print(json.dumps(validation, indent=2))
    print(model_table.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
