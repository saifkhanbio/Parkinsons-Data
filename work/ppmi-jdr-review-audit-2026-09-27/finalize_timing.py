"""Validate completed disjoint timing fits and save local numeric summaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent


def main():
    initial_status=json.loads((ROOT/'timing_status.json').read_text())
    assert initial_status['status']=='COMPLETE'
    for name,expected in json.loads((ROOT/'timing_input_sha256.json').read_text()).items():
        digest=hashlib.sha256()
        with Path(name).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==expected,name
    source=WORK/'ppmi-blood-cell-adjustment-2026-09-12'
    old_hall=pd.read_csv(source/'robust_15_hallmark_comparison.tsv',sep='\t')
    retained=set(old_hall.pathway)
    assert old_hall.pathway.is_unique and retained
    primary_hall=pd.read_csv(WORK/'ppmi-deseq2-2026-09-12/primary/hallmark_enrichment.tsv',sep='\t').set_index('pathway')
    original=pd.read_csv(source/'cell_adjusted/results.tsv',sep='\t').set_index('Geneid')
    candidates=original.index[original.padj.lt(.05)]
    assert candidates.is_unique
    names=pd.read_csv(WORK/'ppmi-five-gene-review-2026-09-12/five_gene_review.tsv',sep='\t').set_index('Geneid')
    rows,genes,pathways=[],[],[]
    all_ids=[]
    for stratum in ['same_month','preceding_months']:
        meta=pd.read_csv(ROOT/stratum/'blood_covariates.tsv',sep='\t',dtype={'PATNO':str})
        assert meta.PATNO.is_unique
        all_ids.extend(meta.PATNO)
        rank=json.loads((ROOT/stratum/'preflight.json').read_text())
        for model in ['subset_reference','cell_adjusted']:
            folder=ROOT/stratum/model
            fit=pd.read_csv(folder/'results.tsv',sep='\t').set_index('Geneid')
            assert len(fit)==21888 and fit.index.is_unique and set(fit.index)==set(original.index)
            assert fit.loc[~fit.beta_converged,'pvalue'].isna().all()
            valid=fit.pvalue.notna()
            np.testing.assert_allclose(multipletests(fit.loc[valid,'pvalue'],method='fdr_bh')[1],
                fit.loc[valid,'padj_no_independent_filter'],rtol=1e-9,atol=1e-12)
            adjusted=fit.padj.notna()
            np.testing.assert_allclose(multipletests(fit.loc[adjusted,'pvalue'],method='fdr_bh')[1],
                fit.loc[adjusted,'padj'],rtol=1e-9,atol=1e-12)
            hall=pd.read_csv(folder/'Hallmark_enrichment.tsv',sep='\t').set_index('pathway')
            coverage=pd.read_csv(folder/'Hallmark_coverage.tsv',sep='\t').set_index('pathway')
            assert len(coverage)==50 and coverage.index.is_unique and hall.index.is_unique
            assert set(hall.index)==set(coverage.index[coverage.eligible])
            assert coverage.eligible.equals(coverage.ranked_members.between(15,500))
            np.testing.assert_allclose(multipletests(hall.pval,method='fdr_bh')[1],hall.padj,rtol=1e-9,atol=1e-12)
            h=hall.loc[sorted(retained.intersection(hall.index))]
            retained_n=int((h.padj.lt(.05)&(np.sign(h.NES)==np.sign(primary_hall.loc[h.index,'NES']))).sum())
            reversed_n=int((h.padj.lt(.05)&(np.sign(h.NES)!=np.sign(primary_hall.loc[h.index,'NES']))).sum())
            same=(np.sign(fit.loc[candidates,'log2FoldChange'])==np.sign(original.loc[candidates,'log2FoldChange']))
            rows.append(dict(stratum=stratum,model=model,n=len(meta),PD=int(meta.group.eq('PD').sum()),
                             Control=int(meta.group.eq('Control').sum()),design_rank=rank['models'][model]['rank'],
                             residual_df=rank['models'][model]['residual_df'],
                             gene_FDR05=int(fit.padj.lt(.05).sum()),hallmark_tested=len(hall),hallmark_FDR05=int(hall.padj.lt(.05).sum()),
                             prior_pathways_testable=len(h),prior_pathways_retained=retained_n,prior_pathways_reversed=reversed_n,converged=int(fit.beta_converged.sum()),
                             nonconverged=int((~fit.beta_converged).sum()),
                             candidate_FDR05=int(fit.loc[candidates,'padj'].lt(.05).sum()),
                             candidate_same_direction=int(same.sum())))
            for gene in candidates:
                r=fit.loc[gene]
                record=dict(stratum=stratum,model=model,Geneid=gene,gene_symbol=names.loc[gene,'reference_gene_name'],log2FoldChange=float(r.log2FoldChange),
                            lfcSE=float(r.lfcSE),pvalue=float(r.pvalue),padj=float(r.padj),
                            FC=float(2**r.log2FoldChange),FC_low=float(2**(r.log2FoldChange-1.96*r.lfcSE)),
                            FC_high=float(2**(r.log2FoldChange+1.96*r.lfcSE)))
                genes.append(record)
            for pathway,r in hall.iterrows():
                pathways.append(dict(stratum=stratum,model=model,pathway=pathway,NES=float(r.NES),padj=float(r.padj),
                                     in_prior_pathway_set=pathway in retained,
                                     same_direction_as_primary=bool(np.sign(r.NES)==np.sign(primary_hall.loc[pathway,'NES']))))
    assert len(all_ids)==len(set(all_ids))==528
    full=pd.read_csv(source/'blood_covariates.tsv',sep='\t',dtype={'PATNO':str})
    assert set(all_ids)==set(full.PATNO)
    summary=pd.DataFrame(rows);summary.to_csv(ROOT/'timing_strata_model_summary.tsv',sep='\t',index=False)
    pd.DataFrame(genes).to_csv(ROOT/'timing_strata_candidates.tsv',sep='\t',index=False)
    pd.DataFrame(pathways).to_csv(ROOT/'timing_strata_Hallmark.tsv',sep='\t',index=False)
    output=dict(validation='PASS',models=rows,
                participant_count=len(all_ids),prior_pathway_count=len(retained),
                candidate_count=len(candidates),
                checks=['unique disjoint participant coverage','fixed gene identities',
                        'convergence flags','BH with/without independent filtering',
                        'Hallmark eligibility and BH','source hashes'])
    (ROOT/'timing_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
