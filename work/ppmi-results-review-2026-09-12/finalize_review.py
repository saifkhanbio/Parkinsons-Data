import os
"""Create the scientific review after the added DESeq2 sensitivity completes."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
A=OUT.parent/'ppmi-deseq2-2026-09-12'
os.environ.setdefault('MPLCONFIGDIR',str(A/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def md(frame):
    def fmt(x):
        if pd.isna(x):return '—'
        if isinstance(x,float):return f'{x:.4g}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    deadline=time.monotonic()+86400
    while not (OUT/'model_summary.json').exists():
        if not args.wait:raise FileNotFoundError('Additional sensitivity model is not ready')
        if 'Execution halted' in (OUT/'models.log').read_text() or time.monotonic()>deadline:
            raise RuntimeError('Additional model failed; inspect models.log')
        time.sleep(10)
    summary=json.loads((OUT/'model_summary.json').read_text())
    availability=json.loads((OUT/'confounder_availability.json').read_text())
    cellsummary=json.loads((OUT/'cell_score_summary.json').read_text())
    design=json.loads((OUT/'site_race_design_check.json').read_text())
    assert summary['status']=='COMPLETE' and cellsummary['status']=='COMPLETE' and design['full_rank']
    primary=pd.read_csv(A/'primary/results_annotated.tsv',sep='\t').set_index('Geneid')
    sensitivity=pd.read_csv(A/'gene_sensitivity_comparison.tsv',sep='\t').set_index('Geneid')
    added=pd.read_csv(OUT/'site_race_results.tsv',sep='\t').set_index('Geneid').reindex(primary.index)
    assert added.index.is_unique and len(added)==len(primary)
    assert added.loc[~added.beta_converged,['pvalue','padj']].isna().all().all()
    p=added.loc[added.padj.notna()]
    ordered=p.sort_values('pvalue')
    bh=np.minimum.accumulate((ordered.pvalue.to_numpy()*len(ordered)/np.arange(1,len(ordered)+1))[::-1])[::-1].clip(max=1)
    np.testing.assert_allclose(bh,ordered.padj,atol=1e-12,rtol=1e-8)
    valid=primary.pvalue.notna()&added.pvalue.notna()
    same=np.sign(primary.log2FoldChange)==np.sign(added.log2FoldChange)
    prior_robust=sensitivity.reindex(primary.index).significant_same_direction_all_models
    six_robust=prior_robust&added.padj.lt(.05)&same
    compare={'primary_vs_added_log2FC_correlation':primary.loc[valid,'log2FoldChange'].corr(added.loc[valid,'log2FoldChange']),
             'primary_significant_retained_same_direction':int((primary.padj.lt(.05)&added.padj.lt(.05)&same).sum()),
             'prior_1917_robust_retained_same_direction':int(six_robust.sum()),
             'added_model_nonconverged':summary['nonconverged']}
    genes=primary.loc[prior_robust].copy()
    genes['fold_change_PD_Control']=2**genes.log2FoldChange
    genes['fold_change_CI95_lower']=2**(genes.log2FoldChange-1.96*genes.lfcSE)
    genes['fold_change_CI95_upper']=2**(genes.log2FoldChange+1.96*genes.lfcSE)
    genes['site_race_log2FC']=added.log2FoldChange
    genes['site_race_padj']=added.padj
    genes['significant_same_direction_all_six']=six_robust
    marker_assoc=pd.read_csv(OUT/'robust_gene_marker_associations.tsv',sep='\t').set_index('Geneid')
    genes=genes.join(marker_assoc[['largest_marker_partial_correlation','most_correlated_immune_score','is_MCP_marker']])
    genes.to_csv(OUT/'robust_gene_review.tsv',sep='\t')
    loo=pd.read_csv(OUT/'flagged_gene_leave_one_out.tsv',sep='\t')
    assert len(loo)==21 and loo.Geneid.nunique()==11 and loo.converged.all()
    loo['absolute_change']=loo.delta_log2FC.abs()
    loo['change_in_original_SE_units']=loo.absolute_change/loo.original_SE
    loo['direction_reversal']=np.sign(loo.loo_log2FC)!=np.sign(loo.original_log2FC)
    review=loo.groupby('Geneid').agg(omissions=('PATNO_omitted','size'),max_abs_log2FC_change=('absolute_change','max'),
        max_change_SE_units=('change_in_original_SE_units','max'),any_direction_reversal=('direction_reversal','any'))
    review=review.join(primary[['gene_symbol','log2FoldChange','padj','max_cooks_all_samples']])
    review['robust_in_original_five']=prior_robust
    review['site_race_padj']=added.padj
    review.to_csv(OUT/'flagged_gene_review.tsv',sep='\t')
    pairs=pd.read_csv(OUT/'flagged_gene_sample_pairs.tsv',sep='\t')
    sample_review=pd.read_csv(OUT/'sample_influence_review.tsv',sep='\t')
    qc=pd.read_csv(OUT.parent/'ppmi-expression-qc-2026-09-12/QC_disposition.tsv',sep='\t')
    qc_columns=['PATNO','assigned_counts','low_library','low_detection','low_RIN','low_mapping',
                'extreme_PC','PCT_USABLE_BASES','excluded_independent_QC','disposition']
    sample_review=sample_review.merge(qc[qc_columns],on='PATNO',validate='one_to_one')
    sample_review.to_csv(OUT/'sample_influence_with_prior_QC.tsv',sep='\t',index=False)
    priority_samples=sample_review[sample_review.PATNO.isin(list(map(int,json.loads(Path(os.environ['PPMI_PRIVATE_CONFIG']).read_text())['influence_review_ids'])))]
    singleton=pairs.design_leverage>.99
    singleton_loo=loo[loo.design_leverage>.99]
    # Balance summaries are descriptive; absence of a history mention is not absence of disease.
    d=pd.read_csv(OUT/'confounder_colData.tsv',sep='\t',dtype={'PATNO':str,'site':str})
    clock=d.RNA_collection_clock.str.extract(r'^(\d{1,2}):(\d{2})')
    d['collection_hour']=pd.to_numeric(clock[0])+pd.to_numeric(clock[1])/60
    balances=[]
    for col in ['age_collection_years','RIN','intergenic_percent','collection_hour']:
        a=d.loc[d.group=='PD',col].dropna();b=d.loc[d.group=='Control',col].dropna()
        balances.append(dict(covariate=col,PD_n=len(a),Control_n=len(b),PD_value=a.mean(),Control_value=b.mean(),
                             standardized_difference=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2),kind='mean'))
    for col in ['diabetes_mention','hypertension_mention','lipid_disorder_mention','allergy_asthma_mention','smoking_mention']:
        a=d.loc[d.group=='PD',col].mean();b=d.loc[d.group=='Control',col].mean()
        den=np.sqrt((a*(1-a)+b*(1-b))/2)
        balances.append(dict(covariate=col,PD_n=int((d.group=='PD').sum()),Control_n=int((d.group=='Control').sum()),
                             PD_value=a,Control_value=b,standardized_difference=(a-b)/den if den else 0,kind='fraction with mention'))
    balance=pd.DataFrame(balances);balance.to_csv(OUT/'clinical_covariate_balance.tsv',sep='\t',index=False)
    pd.crosstab(d.site,d.group).to_csv(OUT/'site_by_group.tsv',sep='\t')
    pd.crosstab(d.fasting_status.fillna('missing'),d.group).to_csv(OUT/'fasting_by_group.tsv',sep='\t')
    pd.crosstab(d.race_category,d.group).to_csv(OUT/'self_reported_race_by_group.tsv',sep='\t')
    effects=pd.read_csv(OUT/'immune_score_group_associations.tsv',sep='\t')
    expanded=pd.read_csv(OUT/'immune_scores_site_race_time_adjusted.tsv',sep='\t')
    effect_table=effects[['cell_population','standardized_effect','padj']].merge(
        expanded[['cell_population','standardized_effect','padj']],on='cell_population',suffixes=('_primary_adjustment','_site_race_clock'))
    effect_table.to_csv(OUT/'immune_score_review.tsv',sep='\t',index=False)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    pos=np.arange(len(effects));sd=effects.adjusted_PD_minus_Control/effects.standardized_effect
    ax.errorbar(effects.standardized_effect,pos-.10,xerr=(effects.CI95_upper-effects.adjusted_PD_minus_Control)/sd,
                fmt='o',label='Primary adjustment (95% HC3 CI)',color='#287F9C')
    ax.scatter(expanded.standardized_effect,pos+.10,label='Add site, race category, collection clock',color='#D86738',marker='s')
    ax.set_yticks(pos,effects.cell_population);ax.axvline(0,color='grey',lw=1)
    ax.set_xlabel('Adjusted PD minus Control / score SD (expression proxy)');ax.legend(fontsize=8,loc='best')
    fig.savefig(OUT/'immune_scores.png',dpi=180);fig.savefig(OUT/'immune_scores.pdf');plt.close(fig)
    correlations=pd.read_csv(OUT/'pathway_marker_partial_correlations.tsv',sep='\t',index_col=0)
    fig,ax=plt.subplots(figsize=(12,8),layout='constrained')
    im=ax.imshow(correlations,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(len(correlations.columns)),correlations.columns,rotation=45,ha='right',fontsize=9)
    ax.set_yticks(range(len(correlations)),correlations.index.str.replace('HALLMARK_','').str.replace('_',' '),fontsize=9)
    fig.colorbar(im,ax=ax,label='Partial correlation; shared marker genes removed')
    fig.savefig(OUT/'pathway_cell_correlations.png',dpi=180);fig.savefig(OUT/'pathway_cell_correlations.pdf');plt.close(fig)
    raw=pd.read_csv(OUT.parent/'ppmi-expression-qc-2026-09-12/raw_counts.tsv.gz',sep='\t',index_col=0)
    sf=pd.read_csv(A/'primary/size_factors.tsv',sep='\t',dtype={'PATNO':str}).set_index('PATNO').loc[d.PATNO,'size_factor']
    normalized=np.log2(raw.loc[review.index,d.PATNO].to_numpy()/sf.to_numpy()[None,:]+1)
    fig,axes=plt.subplots(4,3,figsize=(11,11),layout='constrained');rng=np.random.default_rng(20260912)
    groups=(d.group=='PD').to_numpy().astype(int)
    for i,(gene,ax) in enumerate(zip(review.index,axes.flat)):
        jitter=rng.uniform(-.18,.18,len(d)); ax.scatter(groups+jitter,normalized[i],s=5,alpha=.35,c=np.where(groups,'#D86738','#287F9C'))
        flagged=set(pairs.loc[pairs.Geneid==gene,'PATNO'].astype(str));mark=d.PATNO.isin(flagged).to_numpy()
        ax.scatter(groups[mark]+jitter[mark],normalized[i,mark],s=30,facecolors='none',edgecolors='black')
        symbol=primary.loc[gene,'gene_symbol'];ax.set_title(symbol if pd.notna(symbol) else gene,fontsize=9)
        ax.set_xticks([0,1],['Control','PD']);ax.set_ylabel('log2(normalized count + 1)',fontsize=8)
    axes.flat[-1].axis('off')
    fig.savefig(OUT/'flagged_gene_counts.png',dpi=180);fig.savefig(OUT/'flagged_gene_counts.pdf');plt.close(fig)
    pathways=pd.read_csv(A/'hallmark_sensitivity_comparison.tsv',sep='\t')
    pathways=pathways[pathways.significant_same_direction_all_models].copy()
    pathways['largest_abs_immune_partial_r']=correlations.abs().max(axis=1).reindex(pathways.pathway).to_numpy()
    pathways.to_csv(OUT/'robust_pathway_review.tsv',sep='\t',index=False)
    hallmark=pd.read_csv(A/'primary/hallmark_enrichment.tsv',sep='\t').set_index('pathway')
    examples=[]
    for name in ['INTERFERON_ALPHA_RESPONSE','COMPLEMENT','HEME_METABOLISM','OXIDATIVE_PHOSPHORYLATION']:
        members=set(hallmark.loc['HALLMARK_'+name,'leadingEdge'].split(';'))
        subset=genes[genes.ensembl_id.isin(members)&genes.gene_symbol.notna()].copy()
        subset=subset.loc[subset.log2FoldChange.abs().sort_values(ascending=False).index].head(2)
        for gene,row in subset.iterrows():
            examples.append(dict(pathway=name,gene_symbol=row.gene_symbol,Geneid=gene,
                primary_fold_change=row.fold_change_PD_Control,primary_padj=row.padj,
                site_race_padj=row.site_race_padj,retained_all_six=bool(row.significant_same_direction_all_six)))
    examples=pd.DataFrame(examples)
    examples.to_csv(OUT/'pathway_gene_examples.tsv',sep='\t',index=False)
    hashes=json.loads((OUT/'input_sha256.json').read_text())
    for relative,expected in hashes.items():assert hashlib.sha256((OUT.parent/relative).read_bytes()).hexdigest()==expected
    flags_robust=int(review.robust_in_original_five.sum())
    median_logfc=float(genes.log2FoldChange.abs().median())
    thresholds={str(t):int((genes.log2FoldChange.abs()>=np.log2(t)).sum()) for t in [1.2,1.5,2]}
    large=genes.assign(absolute_log2FC=genes.log2FoldChange.abs()).sort_values('absolute_log2FC',ascending=False).head(10)
    (OUT/'review_summary.json').write_text(json.dumps(dict(status='COMPLETE',influence_genes=11,conditional_omissions=len(loo),
        direction_reversals=int(loo.direction_reversal.sum()),largest_conditional_change_log2FC=float(loo.absolute_change.max()),
        singleton_max_abs_conditional_change=float(singleton_loo.absolute_change.max()),robust_flagged_genes=flags_robust,
        robust_median_absolute_log2FC=median_logfc,robust_fold_change_magnitude_counts=thresholds,
        **compare),indent=2)+'\n')
    report='# Local analysis report\n\nConsult the locally generated tables and diagnostics.\n'
    (OUT/'README.md').write_text(report)
    parent=A/'README.md';text=parent.read_text();link='../ppmi-results-review-2026-09-12/README.md'
    if link not in text:parent.write_text(text+f'\n## Scientific review\n\n[Influence, confounding, and robust-results review]({link}).\n')
    print(json.dumps(compare,indent=2))
    print('Scientific review complete.')


if __name__=='__main__':main()
