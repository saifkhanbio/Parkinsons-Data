"""CUDA result validation and sensitivity statistics; optionally await DESeq2."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
MODELS = ["primary", "medication_timing", "phase", "usable_bases", "all_579"]


def bh(values):
    ordered, order = torch.sort(values)
    scaled = ordered * len(values) / torch.arange(1, len(values) + 1, device=values.device, dtype=values.dtype)
    adjusted = torch.cummin(scaled.flip(0), dim=0).values.flip(0).clamp(max=1)
    result = torch.empty_like(values)
    result[order] = adjusted
    return result


def ranks(values):
    ordered, order = torch.sort(values)
    _, counts = torch.unique_consecutive(ordered, return_counts=True)
    ends = counts.cumsum(0).to(values.dtype)
    average_ranks = ends - (counts.to(values.dtype) - 1) / 2
    result = torch.empty_like(values)
    result[order] = torch.repeat_interleave(average_ranks, counts)
    return result


def correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    return ((x * y).sum() / torch.sqrt((x*x).sum() * (y*y).sum())).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    tensor = lambda values: torch.as_tensor(np.asarray(values, dtype=np.float64), device="cuda", dtype=torch.float64)
    # Small numerical fixtures exercise ties and BH monotonicity before use.
    torch.testing.assert_close(bh(tensor([0.01, 0.04, 0.03, 0.002])), tensor([0.02, 0.04, 0.04, 0.008]))
    torch.testing.assert_close(ranks(tensor([4, 1, 1, 3])), tensor([4, 1.5, 1.5, 3]))
    assert abs(correlation(tensor([1, 2, 3]), tensor([3, 2, 1])) + 1) < 1e-12
    print("CUDA numerical checks passed; awaiting model inputs." if args.wait else "CUDA numerical checks passed.", flush=True)
    deadline = time.monotonic() + 24 * 3600
    while not (OUT / "model_summaries.json").exists():
        if not args.wait:
            raise FileNotFoundError("Run DESeq2 first, or pass --wait")
        if "Execution halted" in (OUT / "run.log").read_text() or time.monotonic() > deadline:
            raise RuntimeError("Model runner failed or exceeded deadline; inspect run.log")
        time.sleep(10)
    started = time.monotonic()
    summaries = json.loads((OUT / "model_summaries.json").read_text())
    tables, validation = {}, {}
    for model in MODELS:
        table = pd.read_csv(OUT / model / "results.tsv", sep="\t").set_index("Geneid")
        assert table.index.is_unique
        errors = {}
        for column in ["padj", "padj_no_independent_filter"]:
            subset = table.loc[table[column].notna()]
            observed, expected = bh(tensor(subset.pvalue)), tensor(subset[column])
            torch.testing.assert_close(observed, expected, rtol=1e-8, atol=1e-12)
            errors[column] = (observed-expected).abs().max().item() if len(subset) else 0.0
        validation[model] = errors
        tables[model] = table
    primary = tables["primary"]
    x, px, padjx = tensor(primary.log2FoldChange), tensor(primary.pvalue), tensor(primary.padj)
    primary_sig = padjx < 0.05
    robust = primary_sig.clone()
    comparison = []
    for model in MODELS[1:]:
        current = tables[model]
        assert set(current.index) == set(primary.index)
        current = current.reindex(primary.index)
        y, py, padjy = tensor(current.log2FoldChange), tensor(current.pvalue), tensor(current.padj)
        valid = torch.isfinite(px) & torch.isfinite(py)
        same = torch.sign(x) == torch.sign(y)
        current_sig = padjy < 0.05
        shared = primary_sig & current_sig
        comparable = primary_sig & valid
        robust &= current_sig & same
        mean = lambda values: values.double().mean().item() if values.numel() else None
        comparison.append(dict(model=model, samples=summaries[model]["n"],
            significant_FDR05=int(current_sig.sum().item()), shared_testable_genes=int(valid.sum().item()),
            log2FC_pearson=correlation(x[valid], y[valid]),
            log2FC_spearman=correlation(ranks(x[valid]), ranks(y[valid])),
            direction_agreement=mean(same[valid]), primary_significant_comparable=int(comparable.sum().item()),
            primary_significant_direction_agreement=mean(same[comparable]),
            shared_significant=int(shared.sum().item()),
            shared_significant_same_direction=int((shared & same).sum().item())))
    pd.DataFrame(comparison).to_csv(OUT / "gpu_sensitivity_summary.tsv", sep="\t", index=False)
    pd.DataFrame({"Geneid": primary.index, "significant_same_direction_all_models": robust.cpu().numpy()}).to_csv(
        OUT / "gpu_robust_gene_flags.tsv", sep="\t", index=False)
    torch.cuda.synchronize()
    result = dict(status="PASSED", device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                  dtype="float64", operations=["BH validation", "Pearson correlation", "tie-aware Spearman correlation",
                                               "effect-direction agreement", "cross-model significant-gene intersection"],
                  model_BH_max_absolute_errors=validation, result_gene_order=MODELS,
                  robust_gene_count=int(robust.sum().item()), elapsed_seconds=time.monotonic()-started,
                  scope="CUDA postprocessing; DESeq2 and fgsea inference retain their CPU implementations")
    temp = OUT / "gpu_result_validation.tmp.json"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(OUT / "gpu_result_validation.json")
    print("CUDA postprocessing complete.", flush=True)


if __name__ == "__main__":
    main()
