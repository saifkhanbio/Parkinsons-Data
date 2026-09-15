"""List genes from saved significant pathways without refitting models."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
BASE=WORK/'ppmi-deseq2-2026-09-12'
FILES={'Hallmark':BASE/'primary/hallmark_enrichment.tsv',
       'C2':BASE/'c2_c5bp/C2_enrichment.tsv',
       'C5_BP':BASE/'c2_c5bp/C5_BP_enrichment.tsv'}
GENES=WORK/'ppmi-classifier-2026-09-12/gene_ids.npy'
RANKS=BASE/'primary/enrichment_ranks.tsv'
SOURCES=[*FILES.values(),GENES,RANKS,BASE/'msigdb_hallmark_gobp.rds',
         BASE/'c2_c5bp/C2_gene_sets.rds',
         WORK/'ppmi-hallmark-classifier-2026-09-12/membership.npz']

def hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}

def save(d,name):
    d.to_csv(ROOT/name,sep='\t',index=False)

def main():
    original=hashes()
    raw=np.load(GENES,allow_pickle=True).astype(str)
    stable=[x.split('.')[0] for x in raw]
    assert len(raw)==58780 and len(set(raw))==len(raw)
    lookup=defaultdict(list)
    for i,g in enumerate(stable): lookup[g].append(i)
    ranks=pd.read_csv(RANKS,sep='\t')
    assert len(ranks)==21885 and ranks.ensembl_id.is_unique
    ranked=set(ranks.ensembl_id)
    hallmark_indices=set(np.load(SOURCES[-1],allow_pickle=True)['indices'].tolist())
    assert len(hallmark_indices)==4376
    summaries=[]; all_rows=[]; unions={}
    for label,file in FILES.items():
        res=pd.read_csv(file,sep='\t')
        sig=res[res.padj.lt(.05)].copy()
        assert len(sig)=={'Hallmark':25,'C2':1505,'C5_BP':1009}[label]
        sig['PD_enrichment_direction']=np.where(sig.NES.gt(0),'higher_PD','lower_PD')
        save(sig,f'{label}_significant_pathways.tsv')
        cached=pd.read_csv(ROOT/f'{label}_cached_memberships.tsv.gz',sep='\t',keep_default_na=False)
        assert set(cached.gs_name)==set(sig.pathway)
        pairs=cached[['gs_name','ensembl_gene']].drop_duplicates()
        members={k:set(v.ensembl_gene) for k,v in pairs.groupby('gs_name')}
        names=cached.groupby('ensembl_gene').gene_symbol.agg(lambda x:';'.join(sorted(set(x)-{''}))).to_dict()
        le={r.pathway:set(str(r.leadingEdge).split(';')) for r in sig.itertuples()}
        for r in sig.itertuples():
            assert len(members[r.pathway]&ranked)==r.size,(label,r.pathway)
            assert le[r.pathway]<=members[r.pathway]&ranked,(label,r.pathway)
        database=set(pairs.ensembl_gene)
        mapped={g for g in database if len(lookup[g])==1}
        edges=set().union(*le.values())
        unambiguous_edges=edges&mapped
        if label=='Hallmark': assert {lookup[g][0] for g in mapped}<=hallmark_indices
        exclusion=[]
        for g in sorted(database-mapped):
            exclusion.append({'ensembl_id':g,'gene_symbol':names.get(g,''),
                              'reason':'not_in_raw_annotation' if not lookup[g] else 'ambiguous_raw_mapping',
                              'raw_matches':';'.join(raw[i] for i in lookup[g])})
        save(pd.DataFrame(exclusion,columns=['ensembl_id','gene_symbol','reason','raw_matches']),f'{label}_mapping_exclusions.tsv')
        long=pairs[pairs.ensembl_gene.isin(mapped)].rename(columns={'gs_name':'pathway','ensembl_gene':'ensembl_id'})
        long=long.merge(sig[['pathway','NES','padj','PD_enrichment_direction']],on='pathway',validate='many_to_one')
        long['Geneid']=[raw[lookup[g][0]] for g in long.ensembl_id]
        long['gene_symbol']=long.ensembl_id.map(names)
        long['in_primary_enrichment_ranking']=long.ensembl_id.isin(ranked)
        long['leading_edge']=[g in le[p] for g,p in zip(long.ensembl_id,long.pathway)]
        long=long[['Geneid','ensembl_id','gene_symbol','pathway','NES','padj','PD_enrichment_direction','in_primary_enrichment_ranking','leading_edge']]
        save(long.sort_values(['pathway','ensembl_id']),f'{label}_gene_pathway_membership.tsv.gz')
        rows=[]
        for g,d in long.groupby('ensembl_id'):
            rows.append({'Geneid':d.Geneid.iloc[0],'ensembl_id':g,'gene_symbol':names[g],
                         'n_significant_pathways':len(d),'n_leading_edge_pathways':int(d.leading_edge.sum()),
                         'in_primary_enrichment_ranking':g in ranked,
                         'n_higher_PD_pathways':int(d.NES.gt(0).sum()),
                         'n_lower_PD_pathways':int(d.NES.lt(0).sum())})
        genes=pd.DataFrame(rows).sort_values(['gene_symbol','ensembl_id'])
        assert genes.Geneid.is_unique and len(genes)==len(mapped)
        save(genes,f'{label}_genes.tsv')
        save(genes[genes.in_primary_enrichment_ranking],f'{label}_ranked_genes.tsv')
        leading=genes[genes.n_leading_edge_pathways.gt(0)]
        assert set(leading.ensembl_id)==unambiguous_edges
        save(leading,f'{label}_leading_edge_genes.tsv')
        # Exact identifiers accompany names; a plain symbol list collapses aliases.
        (ROOT/f'{label}_gene_ids.txt').write_text('\n'.join(sorted(genes.Geneid))+'\n')
        symbols=sorted({s for v in genes.gene_symbol for s in v.split(';') if s})
        (ROOT/f'{label}_gene_symbols.txt').write_text('\n'.join(symbols)+'\n')
        unions[label]=mapped
        gcopy=genes.copy();gcopy['collection']=label;all_rows.append(gcopy)
        summaries.append({'collection':label,'significant_pathways':len(sig),
                          'unique_mapped_member_genes':len(mapped),
                          'members_in_primary_ranking':len(mapped&ranked),
                          'unique_mapped_leading_edge_genes':len(unambiguous_edges),
                          'database_member_ids':len(database),
                          'unmapped_member_ids':sum(not lookup[g] for g in database),
                          'ambiguous_member_ids':sum(len(lookup[g])>1 for g in database),
                          'leading_edge_ids_excluded_as_ambiguous':len(edges-mapped)})
    combined=pd.concat(all_rows,ignore_index=True)
    union=combined.groupby(['Geneid','ensembl_id'],as_index=False).agg(
        gene_symbol=('gene_symbol',lambda x:';'.join(sorted({s for v in x for s in v.split(';') if s}))),
        collections=('collection',lambda x:';'.join(sorted(x))),
        n_significant_pathways=('n_significant_pathways','sum'),
        n_leading_edge_pathways=('n_leading_edge_pathways','sum'),
        in_primary_enrichment_ranking=('in_primary_enrichment_ranking','first'))
    save(union.sort_values(['gene_symbol','ensembl_id']),'combined_unique_genes.tsv')
    save(pd.DataFrame(summaries),'summary.tsv')
    overlap=[{'collection_A':a,'collection_B':b,'shared_mapped_genes':len(unions[a]&unions[b])}
             for a in unions for b in unions if a<b]
    save(pd.DataFrame(overlap),'collection_overlaps.tsv')
    assert hashes()==original
    (ROOT/'input_sha256.json').write_text(json.dumps(original,indent=2)+'\n')
    summary={'model':'primary_558','pathway_FDR':'within_collection_BH_lt_0.05',
             'collections':summaries,'combined_unique_mapped_genes':len(union),
             'checks':'PASS','source_hashes_unchanged':True,'models_refitted':False}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Genes in significant primary PD pathways','',
            'Saved pathway FDR <0.05, both NES directions; 558-person primary model.',
            'Counts below deduplicate unambiguous raw-annotation gene IDs. No new gene filter is applied.','',
            '| Collection | Significant sets | Mapped members | In primary ranking | Leading-edge genes |',
            '|---|---:|---:|---:|---:|']
    for s in summaries:
        report.append(f'| {s["collection"]} | {s["significant_pathways"]:,} | {s["unique_mapped_member_genes"]:,} | {s["members_in_primary_ranking"]:,} | {s["unique_mapped_leading_edge_genes"]:,} |')
    report += ['',f'Combined union: **{len(union):,} genes**; collection totals overlap.','',
               'Main lists: Hallmark_genes.tsv, C2_genes.tsv and C5_BP_genes.tsv.',
               'Each contains exact raw Geneid, stable Ensembl ID, symbol and pathway/leading-edge counts.',
               'Separate ranked-member and leading-edge lists, pathway FDR/NES tables and long membership evidence are included.','',
               'Significance belongs to pathways, not necessarily every member gene. These full-cohort-derived lists are descriptive and should not be fixed feature screens before cross-validation in the same cohort.',
               'No models or enrichment tests were rerun; all source hashes remained unchanged.']
    (ROOT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    with zipfile.ZipFile(ROOT/'significant_pathway_gene_lists.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.iterdir()):
            if p.suffix in {'.tsv','.txt','.json','.md','.gz','.py','.R'}:
                z.write(p,p.name)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
