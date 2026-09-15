"""Validate collection FDRs on CUDA, add joint FDR, and summarize enrichment."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from gpu_postprocess import bh


def main():
    assert torch.cuda.is_available(), "CUDA required for requested GPU postprocessing"
    metadata = pd.read_csv(OUT/"pathway_metadata.tsv",sep="\t")
    provenance = json.loads((OUT/"provenance.json").read_text())
    tables = []
    errors = {}
    for label in ["C2","C5_BP"]:
        d = pd.read_csv(OUT/f"{label}_enrichment.tsv",sep="\t")
        assert d.pathway.is_unique and d['size'].between(15,500).all()
        valid = d.pval.notna()
        assert d.loc[valid,'pval'].between(0,1).all()
        p = torch.tensor(d.loc[valid,'pval'].to_numpy(),device="cuda",dtype=torch.float64)
        # Match p.adjust's default family size even if a pathway p value is unavailable.
        calculated = (bh(p) * len(d)/len(p)).clamp(max=1)
        expected = torch.tensor(d.loc[valid,'padj'].to_numpy(),device="cuda",dtype=torch.float64)
        torch.testing.assert_close(calculated,expected,rtol=1e-8,atol=1e-12)
        errors[label] = (calculated-expected).abs().max().item()
        d.insert(0,"collection",label)
        info = metadata[metadata.collection_label==label].drop(columns="collection_label")
        d = d.merge(info,left_on="pathway",right_on="gs_name",how="left",validate="one_to_one").drop(columns="gs_name")
        assert d.gs_collection.notna().all()
        tables.append(d)
    combined = pd.concat(tables,ignore_index=True)
    valid = combined.pval.notna()
    p = torch.tensor(combined.loc[valid,'pval'].to_numpy(),device="cuda",dtype=torch.float64)
    adjusted = (bh(p)*len(combined)/len(p)).clamp(max=1).cpu().numpy()
    combined['padj_joint_C2_C5_BP'] = np.nan
    combined.loc[valid,'padj_joint_C2_C5_BP'] = adjusted
    combined['direction'] = np.where(combined.NES>0,'higher_in_PD','lower_in_PD')
    combined.to_csv(OUT/"C2_C5_BP_all_results.tsv",sep="\t",index=False)
    combined[combined.padj<0.05].to_csv(OUT/"significant_within_collection_FDR05.tsv",sep="\t",index=False)
    combined[combined.padj_joint_C2_C5_BP<0.05].to_csv(OUT/"significant_joint_FDR05.tsv",sep="\t",index=False)
    summaries = []
    for label,d in combined.groupby('collection',sort=False):
        significant = d.padj<0.05
        summaries.append(dict(collection=label,tested=len(d),unavailable_pvalues=int(d.pval.isna().sum()),
                              significant_FDR05=int(significant.sum()),higher_in_PD=int((significant&(d.NES>0)).sum()),
                              lower_in_PD=int((significant&(d.NES<0)).sum()),
                              significant_joint_FDR05=int((d.padj_joint_C2_C5_BP<0.05).sum())))
    hashes = {str(p.relative_to(OUT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [OUT.parent/'primary/enrichment_ranks.tsv',OUT.parent/'primary/results.tsv',
               OUT.parent/'primary/gobp_enrichment.tsv',OUT/'C2_gene_sets.rds',OUT/'C2_enrichment.tsv',
               OUT/'C5_BP_enrichment.tsv']}
    assert hashes['primary/gobp_enrichment.tsv']==hashes['c2_c5bp/C5_BP_enrichment.tsv']
    validation = dict(status='COMPLETE',gpu=torch.cuda.get_device_name(0),BH_max_absolute_errors=errors,
                      collections=summaries,sha256=hashes,C5_BP_original_preserved=True)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.tsv',sep='\t',index=False)
    lines = ['# C2 and C5 biological-process enrichment','',
             f"Primary PD-versus-Control model; MSigDB {provenance['db_version']}; 21,885 ranked genes.",'',
             '## Results','',
             '| Collection | Tested | FDR <0.05 | Positive NES | Negative NES | Joint FDR <0.05 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        lines.append(f"| {r['collection']} | {r['tested']} | {r['significant_FDR05']} | {r['higher_in_PD']} | {r['lower_in_PD']} | {r['significant_joint_FDR05']} |")
    lines += ['', '## Methods and scope','',
      'C2 includes the full curated collection: chemical/genetic perturbation signatures and canonical pathways. C5_BP denotes C5:GO:BP; it excludes GO molecular function, cellular component, and HPO. These definitions follow [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp).', '',
      'C2 was newly fitted. C5_BP had already been computed as GO biological-process enrichment; its ranked genes, collection membership, tested set sizes, and byte-identical export were verified. No DESeq2 models were refitted. This extension evaluates the primary model only.', '',
      'The unchanged finite, converged primary Wald statistics were used, including genes removed only by independent filtering. Three nonconverged genes were omitted. fgsea multilevel settings: measured set sizes 15–500, seed 20260912, sampleSize=101, nPermSimple=10000, eps=0. Standard fgsea inference runs on CPU; CUDA float64 validates within-collection BH adjustments and computes the additional joint adjustment across C2 and C5_BP.', '',
      'The main FDR is BH <0.05 separately within each complete collection. `padj_joint_C2_C5_BP` additionally controls the combined requested testing family; it does not include prior Hallmark analyses. C2 subcollections are annotated, but FDR was not recalculated separately for each subcollection.', '',
      'Positive NES indicates enrichment toward higher expression in PD; negative NES indicates lower expression. C2 UP/DN names describe the source perturbation signature, not the direction of this PD comparison. Gene-set overlap, inter-gene correlation, and unadjusted blood-cell composition limit biological interpretation; results do not establish causal pathway activation.', '',
      '## Leading sets (smallest within-collection adjusted p values)','']
    for label,d in combined.groupby('collection',sort=False):
        lines += [f'### {label}','','| Gene set | NES | FDR |','| --- | ---: | ---: |']
        for row in d.sort_values('padj').head(10).itertuples():
            lines.append(f'| {row.pathway} | {row.NES:.3f} | {row.padj:.3g} |')
        lines.append('')
    lines += ['## Files and reproduction','',
      '- [Complete results](C2_C5_BP_all_results.tsv), including subcollection, descriptions, URLs, leading-edge genes, and both FDR columns.',
      '- [Within-collection significant sets](significant_within_collection_FDR05.tsv).',
      '- [Joint-FDR significant sets](significant_joint_FDR05.tsv).',
      '- `C2_coverage.tsv` and `C5_BP_coverage.tsv`: every available set and its measured size.',
      '- `provenance.json`, `validation.json`, and `R_session_info.txt`: methods, checksums, and software versions.','',
      '```bash','source "$PPMI_PYTHON_ENV/bin/activate"',
      'Rscript work/ppmi-deseq2-2026-09-12/c2_c5bp/run_enrichment.R',
      'python work/ppmi-deseq2-2026-09-12/c2_c5bp/finalize_gpu.py','```','',
      'CUDA access is required for the final command. Gene sets reuse the versioned local msigdbr cache.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
