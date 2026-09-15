"""Create a transparent, descriptive shortlist from seven-model robust genes."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parent / 'ppmi-deseq2-2026-09-12'
REVIEW = OUT.parent / 'ppmi-results-review-2026-09-12'
QC = OUT.parent / 'ppmi-qc-sensitivity-2026-09-12'
MODELS = ['primary','medication_timing','phase','usable_bases','all_579','site_race','QC']
os.environ.setdefault('MPLCONFIGDIR', str(ANALYSIS / 'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def markdown(frame):
    def value(x):
        if pd.isna(x):
            return '—'
        return f'{x:.4g}' if isinstance(x, float) else str(x).replace('|', '/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |'] +
                     ['| '+' | '.join(value(x) for x in row)+' |'
                      for row in frame.itertuples(index=False,name=None)])


def main():
    assert torch.cuda.is_available(), 'CUDA required'
    devices = [i for i in range(torch.cuda.device_count()) if 'RTX A3000' in torch.cuda.get_device_name(i)]
    assert devices, 'Requested NVIDIA RTX A3000 unavailable'
    torch.cuda.set_device(devices[0])
    tensor = lambda x: torch.as_tensor(np.asarray(x,dtype=np.float64),device='cuda',dtype=torch.float64)
    sources = {}
    def read(path, **kwargs):
        data = Path(path).read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return pd.read_csv(path,sep='\t',**kwargs)

    g = read(QC / 'robust_genes_all_seven.tsv').set_index('Geneid')
    assert len(g) == 1852 and g.index.is_unique and g.robust_all_seven.all()
    assert g.ensembl_id.is_unique
    effects, errors, qvalues, long = [], [], [], []
    for model in MODELS:
        path = (ANALYSIS / model / 'results.tsv' if model not in ['site_race','QC'] else
                REVIEW / 'site_race_results.tsv' if model == 'site_race' else QC / 'results.tsv')
        table = read(path).set_index('Geneid').loc[g.index]
        assert table.beta_converged.all() and table.padj.lt(.05).all()
        assert np.isfinite(table[['log2FoldChange','lfcSE','padj']]).all().all()
        assert table.lfcSE.gt(0).all()
        effects.append(table.log2FoldChange.to_numpy())
        errors.append(table.lfcSE.to_numpy())
        qvalues.append(table.padj.to_numpy())
        long.append(table[['log2FoldChange','lfcSE','padj']].assign(model=model).reset_index())
    betas, ses, qs = np.column_stack(effects), np.column_stack(errors), np.column_stack(qvalues)
    assert (np.sign(betas) == np.sign(betas[:,[0]])).all()
    b, se, q = tensor(betas), tensor(ses), tensor(qs)
    abs_b = b.abs()
    minimum = abs_b.min(1).values.cpu().numpy()
    maximum_q = q.max(1).values.cpu().numpy()
    range_b = (b.max(1).values-b.min(1).values).cpu().numpy()
    # Descriptive marginal CI distance; not a simultaneous seven-model interval.
    ci_distance = (abs_b-1.96*se).min(1).values.cpu().numpy()
    np.testing.assert_allclose(minimum,np.abs(betas).min(1),rtol=0,atol=1e-12)
    np.testing.assert_allclose(maximum_q,qs.max(1),rtol=0,atol=1e-12)
    np.testing.assert_allclose(range_b,np.ptp(betas,axis=1),rtol=0,atol=1e-12)
    np.testing.assert_allclose(ci_distance,(np.abs(betas)-1.96*ses).min(1),rtol=0,atol=1e-12)
    result = g[['gene_symbol','ensembl_id']].copy()
    result['direction'] = np.where(betas[:,0]>0,'higher_in_PD','lower_in_PD')
    result['primary_log2FC'] = betas[:,0]
    result['primary_fold_change_PD_Control'] = 2**betas[:,0]
    result['primary_FC_CI95_lower'] = 2**(betas[:,0]-1.96*ses[:,0])
    result['primary_FC_CI95_upper'] = 2**(betas[:,0]+1.96*ses[:,0])
    result['minimum_abs_log2FC_seven'] = minimum
    result['minimum_fold_change_magnitude_seven'] = 2**minimum
    result['maximum_padj_seven'] = maximum_q
    result['log2FC_range_seven'] = range_b
    result['minimum_marginal_CI_distance_from_zero'] = ci_distance
    pd.concat(long,ignore_index=True).to_csv(OUT / 'seven_model_gene_evidence.tsv',sep='\t',index=False)
    markers = read(REVIEW / 'robust_gene_marker_associations.tsv').set_index('Geneid').loc[g.index]
    result['marker_partial_r'] = markers.largest_marker_partial_correlation
    result['max_abs_marker_partial_r'] = markers.largest_marker_partial_correlation.abs()
    result['most_correlated_immune_score'] = markers.most_correlated_immune_score
    result['is_MCP_marker'] = markers.is_MCP_marker
    assert result.max_abs_marker_partial_r.between(0,1).all()
    flags = read(REVIEW / 'flagged_gene_review.tsv').set_index('Geneid')
    result['primary_Cook_flagged'] = result.index.isin(flags.index)
    result['conditional_max_abs_log2FC_change'] = flags.max_abs_log2FC_change.reindex(result.index)
    result['conditional_max_change_SE_units'] = flags.max_change_SE_units.reindex(result.index)
    result['QC_refit_delta_log2FC'] = g.delta_log2FC
    annotation = read(OUT.parent / 'ppmi-expression-qc-2026-09-12/gene_annotation.tsv').set_index('Geneid')
    result['counted_feature_length'] = annotation.Length.reindex(result.index)

    # Require recurring leading-edge membership and matching gene/pathway direction.
    stable_to_gene = dict(zip(result.ensembl_id,result.index))
    stable_ids = set(stable_to_gene)
    directions = dict(zip(result.ensembl_id,np.sign(betas[:,0])))
    links = []
    hallmark = read(QC / 'hallmark_seven_model_comparison.tsv')
    retained_h = hallmark[hallmark.significant_same_direction_all_seven]
    assert len(retained_h) == 15
    htables = {}
    for model in MODELS:
        path = (ANALYSIS / model / 'hallmark_enrichment.tsv' if model not in ['site_race','QC'] else
                REVIEW / 'site_race_pathways/Hallmark_enrichment.tsv' if model == 'site_race' else QC / 'Hallmark_enrichment.tsv')
        htables[model] = read(path).set_index('pathway')
    for row in retained_h.itertuples():
        members = [set(htables[m].loc[row.pathway,'leadingEdge'].split(';')) for m in MODELS]
        common = set.intersection(*members) & stable_ids
        max_q = max(float(htables[m].loc[row.pathway,'padj']) for m in MODELS)
        for stable in sorted(common):
            if directions[stable] == np.sign(row.primary_NES):
                links.append(dict(Geneid=stable_to_gene[stable],collection='Hallmark',subcollection='',
                                  pathway=row.pathway,models_with_shared_leading_edge=7,
                                  largest_pathway_padj=max_q,description='',source_url=''))
    paths = read(QC / 'pathway_comparison.tsv')
    retained = paths[paths.collection.isin(['C2','C5_BP']) & paths.retained_primary_site_race_QC]
    for row in retained.itertuples():
        members = [set(getattr(row,f'leadingEdge_{m}').split(';')) for m in ['primary','site_race','QC']]
        for stable in sorted(set.intersection(*members) & stable_ids):
            if directions[stable] == np.sign(row.NES_primary):
                links.append(dict(Geneid=stable_to_gene[stable],collection=row.collection,
                    subcollection=row.gs_subcollection,pathway=row.pathway,models_with_shared_leading_edge=3,
                    largest_pathway_padj=max(row.padj_primary,row.padj_site_race,row.padj_QC),
                    description=row.gs_description,source_url=row.gs_url))
    evidence = pd.DataFrame(links)
    assert not evidence.duplicated(['Geneid','collection','pathway']).any()
    evidence['gene_symbol'] = evidence.Geneid.map(result.gene_symbol)
    evidence.to_csv(OUT / 'gene_pathway_evidence.tsv',sep='\t',index=False)
    for collection, column in [('Hallmark','shared_Hallmark_leading_edges_seven'),
                               ('C2','shared_C2_leading_edges_three'),('C5_BP','shared_C5_BP_leading_edges_three')]:
        counts = evidence[evidence.collection == collection].groupby('Geneid').size()
        result[column] = counts.reindex(result.index,fill_value=0)
    # A single representative annotation is for readability, not a ranking weight.
    evidence['display_order'] = np.where(evidence.collection.eq('Hallmark'),0,
        np.where(evidence.collection.eq('C2') & evidence.subcollection.fillna('').str.startswith('CP'),1,
                 np.where(evidence.collection.eq('C5_BP'),2,3)))
    representative = evidence.sort_values(['display_order','largest_pathway_padj','pathway']).drop_duplicates('Geneid').set_index('Geneid')
    result['representative_pathway'] = representative.pathway.reindex(result.index).fillna('No recurring leading-edge support in tested collections')
    result['representative_pathway_models'] = representative.models_with_shared_leading_edge.reindex(result.index)
    result = result.sort_values(['minimum_abs_log2FC_seven','maximum_padj_seven','ensembl_id'],ascending=[False,True,True])
    result['effect_rank_all_1852'] = np.arange(1,len(result)+1)
    named = result[result.gene_symbol.notna() & result.gene_symbol.str.strip().ne('')]
    effect_leaders = named.head(12)
    additional = named[named.max_abs_marker_partial_r.lt(.30) & ~named.index.isin(effect_leaders.index)].head(8)
    assert len(effect_leaders) == 12 and len(additional) == 8
    shortlist = pd.concat([effect_leaders.assign(selection_reason='effect_leader'),
                           additional.assign(selection_reason='additional_lower_marker_correlation')])
    assert shortlist.index.is_unique and len(shortlist) == 20
    result['shortlisted'] = result.index.isin(shortlist.index)
    result.to_csv(OUT / 'all_1852_ranked_genes.tsv',sep='\t')
    shortlist.to_csv(OUT / 'shortlist_20.tsv',sep='\t')
    evidence[evidence.Geneid.isin(shortlist.index)].to_csv(OUT / 'shortlist_pathway_evidence.tsv',sep='\t',index=False)
    unnamed = result[result.gene_symbol.isna()]
    unnamed.to_csv(OUT / 'annotation_review_queue.tsv',sep='\t')
    result[result.primary_Cook_flagged | result.is_MCP_marker].to_csv(OUT / 'influence_and_marker_review.tsv',sep='\t')
    sensitivity = []
    for cutoff in [.20,.30,.40]:
        eligible = named[named.max_abs_marker_partial_r.lt(cutoff) & ~named.index.isin(effect_leaders.index)]
        top = eligible.head(8)
        sensitivity.append(dict(max_abs_r_cutoff=cutoff,eligible_named_additional_genes=len(eligible),
            selected_genes=';'.join(top.gene_symbol),overlap_with_default_eight=len(set(top.index)&set(additional.index))))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / 'marker_cutoff_sensitivity.tsv',sep='\t',index=False)

    fig, ax = plt.subplots(figsize=(9,5.5),layout='constrained')
    ax.scatter(result.max_abs_marker_partial_r,result.minimum_abs_log2FC_seven,s=9,alpha=.3,color='#8095a3',label='All 1,852 robust genes')
    ax.scatter(effect_leaders.max_abs_marker_partial_r,effect_leaders.minimum_abs_log2FC_seven,color='#d46932',s=36,label='12 named effect leaders')
    ax.scatter(additional.max_abs_marker_partial_r,additional.minimum_abs_log2FC_seven,color='#23699d',s=36,marker='s',label='8 additional genes with |r| <0.30')
    ax.axvline(.30,color='grey',ls='--',lw=1)
    ax.set(xlabel='Maximum absolute partial correlation with eight immune-marker scores',
           ylabel='Minimum absolute log2 fold change across seven models')
    ax.legend(fontsize=8)
    fig.savefig(OUT / 'prioritization.png',dpi=180)
    fig.savefig(OUT / 'prioritization.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,9),layout='constrained')
    for i,(gene,row) in enumerate(shortlist.iterrows()):
        j = g.index.get_loc(gene)
        ax.plot([betas[j].min(),betas[j].max()],[i,i],color='#687985',lw=2)
        ax.errorbar(betas[j,0],i,xerr=1.96*ses[j,0],fmt='o',markersize=4,color='#d46932' if i<12 else '#23699d',alpha=.85)
    ax.set_yticks(range(20),shortlist.gene_symbol)
    ax.invert_yaxis(); ax.axvline(0,color='grey',lw=1)
    ax.set_xlabel('PD vs Control log2 fold change; primary Wald 95% CI')
    ax.set_title('Grey segments: observed effect range across seven models',fontsize=11)
    fig.savefig(OUT / 'shortlist_effects.png',dpi=180)
    fig.savefig(OUT / 'shortlist_effects.pdf')
    plt.close(fig)

    def display(frame):
        out = frame[['gene_symbol','primary_fold_change_PD_Control','minimum_fold_change_magnitude_seven',
                     'maximum_padj_seven','max_abs_marker_partial_r']].copy()
        out.columns = ['Gene','Primary PD/Control FC','Minimum FC magnitude (7)','Largest FDR (7)','Max absolute partial r']
        return markdown(out)
    coverage = evidence.groupby('collection').agg(genes=('Geneid','nunique'),pathways=('pathway','nunique')).reset_index()
    stats = dict(status='COMPLETE',gpu=torch.cuda.get_device_name(torch.cuda.current_device()),
        robust_genes=len(result),named_genes=len(named),missing_names_in_current_mapping=len(unnamed),shortlist_genes=20,
        effect_leaders=effect_leaders.gene_symbol.tolist(),additional_lower_marker_correlation=additional.gene_symbol.tolist(),
        robust_genes_max_abs_marker_r_below_0_3=int(result.max_abs_marker_partial_r.lt(.3).sum()),
        direct_MCP_markers_in_shortlist=shortlist[shortlist.is_MCP_marker].gene_symbol.tolist(),
        Cook_flagged_genes_in_shortlist=shortlist[shortlist.primary_Cook_flagged].gene_symbol.tolist(),
        GPU_CPU_summary_validation='PASS at 1e-12 absolute tolerance',no_new_models_or_pvalues=True,input_sha256=sources)
    notes = shortlist[shortlist.is_MCP_marker | shortlist.primary_Cook_flagged][['gene_symbol','is_MCP_marker','primary_Cook_flagged','conditional_max_abs_log2FC_change']]
    report = '# Local analysis report\n\nConsult the locally generated tables and diagnostics.\n'
    (OUT / 'README.md').write_text(report)
    for path,expected in sources.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    (OUT / 'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    parent = QC / 'README.md'; text = parent.read_text()
    link = '../ppmi-gene-prioritization-2026-09-12/README.md'
    if link not in text:
        parent.write_text(text+f'\n## Gene prioritization\n\n[Twenty-gene shortlist and complete evidence for 1,852 robust genes]({link}).\n')
    print(json.dumps({k:v for k,v in stats.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
