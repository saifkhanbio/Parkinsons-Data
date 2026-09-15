"""Independent CUDA checks of counts, prefilter, and all five design matrices.

This does not replace or claim to accelerate DESeq2's CPU model fitting.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
started = time.monotonic()
assert torch.cuda.is_available(), "CUDA access is required; no silent CPU fallback"
device = torch.device("cuda:0")
torch.cuda.reset_peak_memory_stats()
raw = pd.read_csv(OUT.parent / "ppmi-expression-qc-2026-09-12/raw_counts.tsv.gz", sep="\t", index_col=0)
counts = torch.tensor(raw.to_numpy(dtype=np.float64), dtype=torch.float64, device=device)
assert counts.shape == (58780, 579)
assert torch.isfinite(counts).all().item()
assert (counts >= 0).all().item()
assert (counts == counts.floor()).all().item()
assert (counts.sum(dim=0) > 0).all().item()
primary = pd.read_csv(OUT / "primary_colData.tsv", sep="\t", dtype={"PATNO": str})
indices = torch.tensor(raw.columns.get_indexer(primary.PATNO), device=device)
assert (indices >= 0).all().item()
sample_counts = (counts.index_select(1, indices) >= 10).sum(dim=1)
prefilter = pd.read_csv(OUT / "gene_prefilter.tsv", sep="\t")
assert raw.index.tolist() == prefilter.Geneid.tolist()
np.testing.assert_array_equal(sample_counts.cpu().numpy(), prefilter.primary_samples_count_at_least_10)
np.testing.assert_array_equal((sample_counts >= 179).cpu().numpy(), prefilter.prefilter_pass)
qc = pd.read_csv(OUT.parent / "ppmi-expression-qc-2026-09-12/sample_QC.tsv", sep="\t", dtype={"PATNO": str}).set_index("PATNO")
np.testing.assert_array_equal(counts.sum(dim=0).cpu().numpy(), qc.loc[raw.columns, "assigned_counts"])
checks = json.loads((OUT / "design_checks.json").read_text())
designs = {}
for name in ["primary", "medication_timing", "phase", "usable_bases", "all_579"]:
    d = pd.read_csv(OUT / f"{name}_colData.tsv", sep="\t")
    batch = "phase" if name == "phase" else "batch"
    quality = "usable_c" if name == "usable_bases" else "intergenic_c"
    columns = [np.ones((len(d), 1)), pd.get_dummies(d[batch], drop_first=True).to_numpy(dtype=float),
               d[["age_c"]].to_numpy(), (d.sex == "Male").to_numpy(dtype=float)[:, None],
               d[["RIN_c", quality]].to_numpy(), (d.group == "PD").to_numpy(dtype=float)[:, None]]
    x = torch.tensor(np.column_stack(columns), dtype=torch.float64, device=device)
    singular_values = torch.linalg.svdvals(x)
    tolerance = max(x.shape) * torch.finfo(x.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    assert rank == x.shape[1] == checks[name]["rank"]
    assert x.shape[0] == checks[name]["n"]
    z = x[:, 1:]
    z = (z-z.mean(dim=0))/z.std(dim=0, correction=1)
    scaled = torch.cat([torch.ones((len(d), 1), device=device, dtype=torch.float64), z], dim=1)
    s = torch.linalg.svdvals(scaled)
    condition = (s.max()/s.min()).item()
    designs[name] = {"samples": len(d), "columns": x.shape[1], "rank_cuda_float64": rank,
                     "scaled_condition_number_cuda_float64": condition}
torch.cuda.synchronize()
result = dict(status="PASSED", device=torch.cuda.get_device_name(0),
              torch_version=torch.__version__, cuda_version=torch.version.cuda,
              operations="count validity, library totals, fixed prefilter, float64 SVD of five designs",
              count_shape=list(counts.shape), retained_genes=int((sample_counts >= 179).sum().item()),
              exact_agreement_with_CPU_count_checks=True, designs=designs,
              peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
              elapsed_seconds=time.monotonic()-started,
              DESeq2_backend="CPU; CUDA validation does not replace the DESeq2 fit")
(OUT / "gpu_validation.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
