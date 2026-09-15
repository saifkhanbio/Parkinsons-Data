"""Freeze outcome-independent annotation mapping and original feature lengths."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ppmi-classifier-2026-09-12'
COLLECTIONS=['Hallmark','C2_CP','C5_BP']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    genes=np.load(BASE/'gene_ids.npy');stable=np.array([g.split('.')[0] for g in genes]);counts=pd.Series(stable).value_counts()
    unique=np.array([counts[g]==1 for g in stable]);lookup={g:i for i,g in enumerate(stable) if unique[i]}
    ann=ROOT.parent/'ppmi-expression-qc-2026-09-12/gene_annotation.tsv';length=pd.read_csv(ann,sep='\t').set_index('Geneid').loc[genes,'Length'].to_numpy(float)
    assert len(genes)==58780 and np.isfinite(length).all() and (length>0).all()
    pd.DataFrame(dict(raw_index=np.arange(len(genes)),Geneid=genes,stable_id=stable,Length=length,unambiguous=unique)).to_csv(ROOT/'gene_reference.tsv',sep='\t',index=False)
    rows=[];cols=[];sets=[];mapped=[];excluded=[];inputs=[BASE/'gene_ids.npy',ann,ROOT/'prepare_mapping.py',ROOT/'export_sets.R',ROOT/'README.md']
    for collection in COLLECTIONS:
        p=ROOT/(collection+'_annotation.tsv.gz');inputs.append(p)
        inputs.append(ROOT.parent/'ppmi-deseq2-2026-09-12/cache/R/msigdbr'/('msigdb.2026.1.Hs.'+{'Hallmark':'H','C2_CP':'C2','C5_BP':'C5'}[collection]+'.rds'))
        t=pd.read_csv(p,sep='\t',dtype=str).fillna('');assert set(t.db_version)=={'2026.1.Hs'}
        if collection=='C2_CP':assert t.gs_subcollection.str.match(r'^CP(?::|$)').all()
        if collection=='C5_BP':assert set(t.gs_subcollection)=={'GO:BP'}
        for name,frame in t.groupby('gs_name',sort=True):
            n=len(sets);ids=sorted(set(frame.db_ensembl_gene));member=[]
            for g in ids:
                if g in lookup:
                    i=lookup[g];member.append(i);mapped.append(dict(collection=collection,pathway=name,Geneid=genes[i],raw_index=i))
                else:excluded.append(dict(collection=collection,pathway=name,stable_id=g,reason='ambiguous_raw_mapping' if g in counts else 'absent_raw_mapping'))
            member=sorted(set(member));rows.extend([n]*len(member));cols.extend(member)
            first=frame.iloc[0];sets.append(dict(pathway_index=n,collection=collection,pathway=name,gs_id=first.gs_id,subcollection=first.gs_subcollection,description=first.gs_description,annotation_members=len(ids),mapped_members=len(member)))
    matrix=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(sets),len(genes)))
    assert matrix.max()==1
    sparse.save_npz(ROOT/'membership.npz',matrix)
    pd.DataFrame(sets).to_csv(ROOT/'pathway_reference.tsv',sep='\t',index=False)
    pd.DataFrame(mapped).to_csv(ROOT/'mapped_membership.tsv.gz',sep='\t',index=False)
    pd.DataFrame(excluded).to_csv(ROOT/'mapping_exclusions.tsv.gz',sep='\t',index=False)
    summary=dict(raw_genes=len(genes),unambiguous_background_candidates=int(unique.sum()),positive_lengths=int((length>0).sum()),pathways=pd.DataFrame(sets).collection.value_counts().to_dict(),mapped_memberships=matrix.nnz,database_version='2026.1.Hs')
    (ROOT/'mapping_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'mapping_input_sha256.json').write_text(json.dumps({str(p):sha(p) for p in inputs},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':prepare()
