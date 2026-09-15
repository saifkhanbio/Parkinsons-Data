"""Validate STRING mappings; calculate network metrics and pathway overlaps."""
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import torch

OUT=Path(__file__).resolve().parent
ANALYSIS=OUT.parent/'ppmi-deseq2-2026-09-12'


def main():
    genes=pd.read_csv(OUT/'input_20_genes.tsv',sep='\t').set_index('ensembl_id')
    raw=json.loads((OUT/'string_network_raw.json').read_text())
    assert len(genes)==20 and genes.index.is_unique
    mapping=[]
    for item in raw['elements']['nodes']:
        d=item['data']; query=d['query_term']
        assert query in genes.index and d['stringdb_species']=='Homo sapiens'
        mapping.append(dict(ensembl_id=query,STRING_id=d['name'],STRING_name=d['display_name'],
                            source_SUID=d['id'],gene_symbol=genes.loc[query,'gene_symbol']))
    mapping=pd.DataFrame(mapping)
    assert mapping.ensembl_id.is_unique and mapping.STRING_id.is_unique and set(mapping.ensembl_id)==set(genes.index)
    mapping.to_csv(OUT/'STRING_mapping.tsv',sep='\t',index=False)
    ids=dict(zip(mapping.source_SUID,mapping.ensembl_id))
    edges=[]
    for item in raw['elements']['edges']:
        d=item['data'];a,b=sorted([ids[d['source']],ids[d['target']]])
        assert a!=b
        edges.append(dict(source=a,target=b,source_gene=genes.loc[a,'gene_symbol'],target_gene=genes.loc[b,'gene_symbol'],
            **{k.removeprefix('stringdb_'):v for k,v in d.items() if k.startswith('stringdb_')}))
    edges=pd.DataFrame(edges)
    assert not edges.duplicated(['source','target']).any() and edges.score.ge(.4).all()
    edges.to_csv(OUT/'STRING_edges_0.4.tsv',sep='\t',index=False)
    metrics=[]
    for cutoff in [.4,.7,.9]:
        graph=nx.Graph();graph.add_nodes_from(genes.index)
        subset=edges[edges.score.ge(cutoff)]
        graph.add_weighted_edges_from(subset[['source','target','score']].itertuples(index=False,name=None))
        components=sorted(nx.connected_components(graph),key=lambda c:(-len(c),sorted(c)))
        centrality=nx.betweenness_centrality(graph,normalized=True,weight=None)
        nodes=genes.copy()
        nodes['STRING_degree']=[graph.degree(n) for n in nodes.index]
        nodes['STRING_betweenness_unweighted']=[centrality[n] for n in nodes.index]
        nodes['STRING_component']=[next(i+1 for i,c in enumerate(components) if n in c) for n in nodes.index]
        nodes['STRING_component_size']=[next(len(c) for c in components if n in c) for n in nodes.index]
        nodes['review_flag']=np.where(nodes.primary_Cook_flagged,'Cook flagged',np.where(nodes.is_MCP_marker,'MCP marker','none'))
        nodes.to_csv(OUT/f'gene_nodes_{cutoff:.1f}.tsv',sep='\t')
        subset.to_csv(OUT/f'gene_edges_{cutoff:.1f}.tsv',sep='\t',index=False)
        metrics.append(dict(cutoff=cutoff,nodes=graph.number_of_nodes(),edges=graph.number_of_edges(),
            isolates=len(list(nx.isolates(graph))),components=len(components),largest_component=len(components[0]),
            density=nx.density(graph),max_degree=max(dict(graph.degree()).values()),
            max_betweenness=max(centrality.values())))
    pd.DataFrame(metrics).to_csv(OUT/'confidence_sensitivity.tsv',sep='\t',index=False)

    paths=pd.read_csv(OUT/'selected_pathways.tsv',sep='\t').set_index('pathway')
    membership=pd.read_csv(OUT/'pathway_membership.tsv',sep='\t')
    ranks=pd.read_csv(ANALYSIS/'primary/enrichment_ranks.tsv',sep='\t')
    ranks[['ensembl_id','stat']].to_csv(OUT/'primary_ranks.rnk',sep='\t',index=False,header=['Gene','Rank'])
    universe=set(ranks.ensembl_id)
    sets={p:set(membership.loc[membership.gs_name.eq(p),'ensembl_gene'])&universe for p in paths.index}
    assert len(sets)==51 and all(len(s)>0 for s in sets.values())
    all_ids=sorted(set.union(*sets.values()))
    lookup={v:i for i,v in enumerate(all_ids)}
    binary=np.zeros((len(paths),len(all_ids)),dtype=np.float64)
    for i,p in enumerate(paths.index):binary[i,[lookup[x] for x in sets[p]]]=1
    assert torch.cuda.is_available()
    devices=[i for i in range(torch.cuda.device_count()) if 'RTX A3000' in torch.cuda.get_device_name(i)]
    assert devices;torch.cuda.set_device(devices[0])
    x=torch.as_tensor(binary,device='cuda',dtype=torch.float64)
    intersection=x@x.T; sizes=x.sum(1)
    jaccard=(intersection/(sizes[:,None]+sizes[None,:]-intersection)).cpu().numpy()
    overlap_edges=[]
    for i,a in enumerate(paths.index):
        for j in range(i+1,len(paths)):
            b=paths.index[j];expected=len(sets[a]&sets[b])/len(sets[a]|sets[b])
            assert abs(jaccard[i,j]-expected)<1e-12
            if jaccard[i,j]>=.25:
                overlap_edges.append(dict(source=a,target=b,jaccard=jaccard[i,j],shared_genes=len(sets[a]&sets[b])))
    pd.DataFrame(overlap_edges).to_csv(OUT/'pathway_overlap_edges.tsv',sep='\t',index=False)
    paths['measured_genes']=[len(sets[p]) for p in paths.index]
    paths['shortlist_members']=[';'.join(genes.loc[sorted(sets[p]&set(genes.index)),'gene_symbol']) for p in paths.index]
    paths['shortlist_member_count']=[len(sets[p]&set(genes.index)) for p in paths.index]
    paths['display_label']=paths.index.str.replace('HALLMARK_','').str.replace('REACTOME_','').str.replace('GOBP_','').str.replace('_',' ')
    paths.to_csv(OUT/'pathway_nodes.tsv',sep='\t')
    results=[]
    files={'Hallmark':ANALYSIS/'primary/hallmark_enrichment.tsv','C2':ANALYSIS/'c2_c5bp/C2_enrichment.tsv',
           'C5_BP':ANALYSIS/'c2_c5bp/C5_BP_enrichment.tsv'}
    for collection,file in files.items():
        current=pd.read_csv(file,sep='\t').set_index('pathway')
        for name,row in paths[paths.collection.eq(collection)].iterrows():
            r=current.loc[name]
            assert abs(r.padj-row.padj_primary)<1e-12
            results.append(dict(Name=name,Description=name,pvalue=r.pval,qvalue=r.padj,
                                Phenotype=1 if r.NES>0 else -1,Genes=','.join(sorted(sets[name]))))
    pd.DataFrame(results).to_csv(OUT/'enrichmentmap_results.tsv',sep='\t',index=False)
    (OUT/'analysis_summary.json').write_text(json.dumps(dict(status='PREPARED',STRING_version=raw['data']['data_version'],
        mapped_genes=len(mapping),confidence_sensitivity=metrics,pathway_nodes=len(paths),
        pathway_edges_Jaccard_0_25=len(overlap_edges),GPU=torch.cuda.get_device_name(torch.cuda.current_device()),
        pathway_overlap_validation='All pairs agree with independent Python set operations to 1e-12',
        new_enrichment_tests=False),indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False))
    print('Pathways:',len(paths),'Overlap edges:',len(overlap_edges))


if __name__=='__main__':main()
