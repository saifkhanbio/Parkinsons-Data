"""CUDA marker scores and covariate-adjusted descriptive association analyses."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

OUT=Path(__file__).resolve().parent
A=OUT.parent/'ppmi-deseq2-2026-09-12'
sys.path.insert(0,str(A))
from gpu_postprocess import bh


def main():
    assert torch.cuda.is_available(), 'CUDA required'
    T=lambda v: torch.as_tensor(np.asarray(v,dtype=np.float64),dtype=torch.float64,device='cuda')
    d=pd.read_csv(OUT/'confounder_colData.tsv',sep='\t',dtype={'PATNO':str})
    raw=pd.read_csv(OUT.parent/'ppmi-expression-qc-2026-09-12/raw_counts.tsv.gz',sep='\t',index_col=0)
    raw=raw[d.PATNO]
    stable=raw.index.str.replace(r'\.[0-9]+$','',regex=True)
    assert not stable.duplicated().any()
    sf=pd.read_csv(A/'primary/size_factors.tsv',sep='\t',dtype={'PATNO':str}).set_index('PATNO').loc[d.PATNO,'size_factor']
    logcounts=torch.log2(T(raw.to_numpy())/T(sf.to_numpy())[None,:]+1)
    markers=pd.read_csv(OUT/'MCPcounter_genes.tsv',sep='\t')
    immune=[p for p in markers['Cell population'].unique() if p not in ['Endothelial cells','Fibroblasts']]
    scores=[];coverage=[];all_marker_ids=set()
    for cell in immune:
        subset=markers[markers['Cell population']==cell]
        ids=subset['ENSEMBL ID'].dropna().unique()
        positions=stable.get_indexer(ids);found=positions>=0
        assert found.sum()>0
        all_marker_ids.update(ids[found])
        score=logcounts[torch.tensor(positions[found],device='cuda')].mean(dim=0)
        # Independently check the published mean-of-log-expression operation on CPU.
        cpu=np.log2(raw.iloc[positions[found]].to_numpy()/sf.to_numpy()[None,:]+1).mean(axis=0)
        np.testing.assert_allclose(score.cpu().numpy(),cpu,atol=1e-10,rtol=1e-10)
        scores.append(score)
        marker_counts=raw.iloc[positions[found]].to_numpy()
        coverage.append(dict(cell_population=cell,published_marker_rows=len(subset),mapped_unique_markers=len(ids),
                             measured_markers=int(found.sum()),markers_at_least_10_counts_in_179_samples=int(((marker_counts>=10).sum(axis=1)>=179).sum()),
                             missing_marker_ids=';'.join(ids[~found])))
    scores=torch.stack(scores,dim=1)
    pd.DataFrame(scores.cpu().numpy(),index=pd.Index(d.PATNO,name='PATNO'),columns=immune).to_csv(OUT/'immune_expression_scores.tsv',sep='\t')
    pd.DataFrame(coverage).to_csv(OUT/'marker_coverage.tsv',sep='\t',index=False)
    design=pd.read_csv(OUT/'primary_design_matrix.tsv',sep='\t',dtype={'PATNO':str}).set_index('PATNO').loc[d.PATNO]
    h=pd.read_csv(A/'primary/sample_influence.tsv',sep='\t',dtype={'PATNO':str}).set_index('PATNO').loc[d.PATNO,'design_leverage']
    keep=h.to_numpy()<0.99
    # Singleton batches carry no within-batch information for the group contrast.
    # Remove these saturated observations for HC3 score analysis, not from DESeq2.
    x=T(design.loc[keep].to_numpy())
    gindex=design.columns.get_loc('groupPD')
    g=x[:,gindex]; nuisance=torch.cat([x[:,:gindex],x[:,gindex+1:]],dim=1)
    u,s,_=torch.linalg.svd(nuisance,full_matrices=False)
    rank=int((s>max(nuisance.shape)*torch.finfo(s.dtype).eps*s.max()).sum())
    u=u[:,:rank]
    gr=g-u@(u.T@g);denom=(gr*gr).sum()
    assert denom>0
    y=scores[torch.as_tensor(keep,device='cuda')]
    beta=(gr@y)/denom
    residual=y-u@(u.T@y)-gr[:,None]*beta[None,:]
    leverage=(u*u).sum(1)+gr*gr/denom
    assert (leverage<0.99).all()
    se=torch.sqrt(((gr[:,None]*residual/(1-leverage[:,None]))**2).sum(0))/denom
    df=len(g)-rank-1
    p=2*student_t.sf(np.abs((beta/se).cpu().numpy()),df=df)
    adj=bh(T(p)).cpu().numpy()
    effects=pd.DataFrame(dict(cell_population=immune,adjusted_PD_minus_Control=beta.cpu().numpy(),
                              HC3_SE=se.cpu().numpy(),pvalue=p,padj=adj,
                              standardized_effect=(beta/y.std(0)).cpu().numpy(),
                              CI95_lower=(beta-student_t.ppf(.975,df)*se).cpu().numpy(),
                              CI95_upper=(beta+student_t.ppf(.975,df)*se).cpu().numpy()))
    effects.to_csv(OUT/'immune_score_group_associations.tsv',sep='\t',index=False)
    # Repeat score comparisons with explicit site/race and collection-clock terms.
    expanded=pd.read_csv(OUT/'site_race_design_matrix.tsv',sep='\t',dtype={'PATNO':str}).set_index('PATNO').loc[d.PATNO]
    clock=d.RNA_collection_clock.str.extract(r'^(\d{1,2}):(\d{2})')
    hours=pd.to_numeric(clock[0])+pd.to_numeric(clock[1])/60
    assert hours.notna().all() and hours.between(0,24,inclusive='left').all()
    expanded['clock_sin']=np.sin(hours.to_numpy()*2*np.pi/24)
    expanded['clock_cos']=np.cos(hours.to_numpy()*2*np.pi/24)
    xe=T(expanded.to_numpy())
    ue,se,_=torch.linalg.svd(xe,full_matrices=False)
    re=int((se>max(xe.shape)*torch.finfo(se.dtype).eps*se.max()).sum())
    retained=(ue[:,:re]**2).sum(1)<.99
    xe=xe[retained];ye=scores[retained]
    gi=expanded.columns.get_loc('groupPD');ge=xe[:,gi]
    ne=torch.cat([xe[:,:gi],xe[:,gi+1:]],dim=1)
    ue,se,_=torch.linalg.svd(ne,full_matrices=False)
    re=int((se>max(ne.shape)*torch.finfo(se.dtype).eps*se.max()).sum());ue=ue[:,:re]
    gre=ge-ue@(ue.T@ge);de=(gre*gre).sum();be=(gre@ye)/de
    ee=ye-ue@(ue.T@ye)-gre[:,None]*be[None,:]
    he=(ue*ue).sum(1)+gre*gre/de
    assert (he<.99).all()
    see=torch.sqrt(((gre[:,None]*ee/(1-he[:,None]))**2).sum(0))/de
    pe=2*student_t.sf(np.abs((be/see).cpu().numpy()),df=len(ge)-re-1)
    alt=pd.DataFrame(dict(cell_population=immune,adjusted_PD_minus_Control=be.cpu().numpy(),
                           HC3_SE=see.cpu().numpy(),pvalue=pe,padj=bh(T(pe)).cpu().numpy(),
                           standardized_effect=(be/ye.std(0)).cpu().numpy(),samples=int(retained.sum())))
    alt.to_csv(OUT/'immune_scores_site_race_time_adjusted.tsv',sep='\t',index=False)
    # Control both the original nuisance variables and diagnosis for partial correlations.
    full=torch.cat([u,(gr/torch.sqrt(denom))[:,None]],dim=1)
    def resid(z): return z-full@(full.T@z)
    def unit(z):
        z=resid(z)
        return z/torch.sqrt((z*z).sum(0,keepdim=True))
    marker_unit=unit(y)
    htable=pd.read_csv(A/'hallmark_sensitivity_comparison.tsv',sep='\t')
    robust_pathways=htable.loc[htable.significant_same_direction_all_models,'pathway']
    memberships=pd.read_csv(OUT/'hallmark_membership.tsv',sep='\t')
    rankids=set(pd.read_csv(A/'primary/enrichment_ranks.tsv',sep='\t').ensembl_id)
    programs=[];program_metadata=[]
    for pathway in robust_pathways:
        ids=set(memberships.loc[memberships.gs_name==pathway,'ensembl_gene'].dropna()) & rankids
        nonmarker=sorted(ids-all_marker_ids)
        pos=stable.get_indexer(nonmarker);assert (pos>=0).all()
        z=logcounts[torch.tensor(pos,device='cuda')][:,torch.as_tensor(keep,device='cuda')]
        variable=z.std(dim=1)>0;z=z[variable]
        z=(z-z.mean(1,keepdim=True))/z.std(1,keepdim=True)
        programs.append(z.mean(0))
        program_metadata.append(dict(pathway=pathway,measured_members=len(ids),marker_members_removed=len(ids&all_marker_ids),
                                      remaining_variable_members=int(variable.sum())))
    programs=torch.stack(programs,dim=1)
    correlations=(unit(programs).T@marker_unit).cpu().numpy()
    pd.DataFrame(correlations,index=pd.Index(robust_pathways,name='pathway'),columns=immune).to_csv(
        OUT/'pathway_marker_partial_correlations.tsv',sep='\t')
    pd.DataFrame(program_metadata).to_csv(OUT/'pathway_score_coverage.tsv',sep='\t',index=False)
    genes=pd.read_csv(A/'gene_sensitivity_comparison.tsv',sep='\t')
    genes=genes[genes.significant_same_direction_all_models].copy()
    pos=stable.get_indexer(genes.ensembl_id);assert (pos>=0).all()
    expr=logcounts[torch.tensor(pos,device='cuda')][:,torch.as_tensor(keep,device='cuda')].T
    corr=(unit(expr).T@marker_unit).cpu().numpy()
    maximum=np.argmax(abs(corr),axis=1)
    genes['largest_marker_partial_correlation']=corr[np.arange(len(genes)),maximum]
    genes['most_correlated_immune_score']=np.asarray(immune)[maximum]
    genes['is_MCP_marker']=genes.ensembl_id.isin(all_marker_ids)
    genes.to_csv(OUT/'robust_gene_marker_associations.tsv',sep='\t',index=False)
    summary=dict(status='COMPLETE',gpu=torch.cuda.get_device_name(0),score_samples=len(d),
                 association_samples=int(keep.sum()),saturated_batch_samples_omitted=d.PATNO[~keep].tolist(),
                 site_race_clock_score_samples=int(retained.sum()),
                 residual_df=df,immune_populations=immune,significant_immune_scores=int((adj<.05).sum()),
                 robust_nonmarker_genes_abs_partial_r_over_0_5=int(((~genes.is_MCP_marker)&(genes.largest_marker_partial_correlation.abs()>.5)).sum()),
                 scores_are_cell_fractions=False,CPU_marker_mean_check='passed at 1e-10 tolerance',
                 no_expression_derived_covariates_added_to_DESeq2=True)
    (OUT/'cell_score_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(effects.to_string(index=False))


if __name__=='__main__':main()
