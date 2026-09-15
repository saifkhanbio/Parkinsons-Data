"""Match robust count features to the exact GENCODE reference in the archive."""
import gzip
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PRIORITY = OUT.parent / 'ppmi-gene-prioritization-2026-09-12'
GTF = OUT / 'cache/gencode.v29.primary_assembly.annotation.gtf.gz'
URL = 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_29/gencode.v29.primary_assembly.annotation.gtf.gz'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def markdown(frame):
    def value(x):
        if pd.isna(x):return '—'
        return f'{x:.4g}' if isinstance(x,float) else str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(value,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(value,row))+' |' for row in frame.itertuples(index=False,name=None)])


def union_length(intervals):
    length = 0
    end = -1
    for start, stop in sorted(set(intervals)):
        if start > end:
            length += stop-start+1
        elif stop > end:
            length += stop-end
        end = max(end,stop)
    return length


def main():
    sources = [PRIORITY/'all_1852_ranked_genes.tsv',PRIORITY/'shortlist_20.tsv',
               PRIORITY/'annotation_review_queue.tsv',GTF,
               OUT.parent/'ppmi-expression-qc-2026-09-12/gene_annotation.tsv']
    hashes = {str(p):sha256(p) for p in sources}
    genes = pd.read_csv(sources[0],sep='\t').set_index('Geneid')
    shortlist = pd.read_csv(sources[1],sep='\t').set_index('Geneid')
    queue = pd.read_csv(sources[2],sep='\t').set_index('Geneid')
    assert len(genes)==1852 and len(shortlist)==20 and len(queue)==525
    assert genes.index.is_unique and genes.ensembl_id.is_unique
    needed = set(genes.index)
    reference_rows, headers, exon_intervals = [], [], defaultdict(lambda:defaultdict(list))
    attributes = re.compile(r'(\w+) "([^"]*)"')
    gene_id = re.compile(r'gene_id "([^"]+)"')
    # Parse the entire stream so gzip CRC/truncation errors cannot pass silently.
    with gzip.open(GTF,'rt') as f:
        for line in f:
            if line.startswith('#'):
                headers.append(line.rstrip());continue
            parts = line.rstrip('\n').split('\t')
            assert len(parts)==9
            if parts[2] not in ['gene','exon']:continue
            match = gene_id.search(parts[8])
            assert match is not None
            identifier = match.group(1)
            if parts[2]=='gene':
                a = dict(attributes.findall(parts[8]))
                reference_rows.append(dict(reference_Geneid=identifier,
                    ensembl_id=re.sub(r'\.[0-9]+$','',identifier),reference_gene_name=a.get('gene_name',''),
                    reference_gene_type=a.get('gene_type',''),reference_gene_status=a.get('gene_status',''),
                    reference_gene_level=a.get('level',''),chromosome=parts[0],start=int(parts[3]),end=int(parts[4]),
                    strand=parts[6],annotation_source=parts[1]))
            elif identifier in needed:
                exon_intervals[identifier][(parts[0],parts[6])].append((int(parts[3]),int(parts[4])))
    reference = pd.DataFrame(reference_rows).set_index('reference_Geneid')
    assert reference.index.is_unique
    reference.to_csv(OUT/'gencode_v29_gene_reference.tsv',sep='\t')
    (OUT/'reference_header.txt').write_text('\n'.join(headers)+'\n')
    by_stable = defaultdict(list)
    for identifier,stable in reference.ensembl_id.items():by_stable[stable].append(identifier)
    mappings = []
    for identifier,row in genes.iterrows():
        exact = identifier in reference.index
        candidates = [identifier] if exact else by_stable.get(row.ensembl_id,[])
        if len(candidates)==1:
            ref_id = candidates[0]; ann = reference.loc[ref_id].to_dict()
            ann.pop('ensembl_id')
            status = 'exact_version' if exact else 'stable_id_different_version'
        else:
            ref_id = '';ann = {}
            status = 'unmapped' if len(candidates)==0 else 'ambiguous_stable_id'
        length = (sum(union_length(v) for v in exon_intervals[identifier].values())
                  if exact and identifier in exon_intervals else np.nan)
        mappings.append(dict(Geneid=identifier,reference_Geneid=ref_id,match_status=status,
            exon_union_length=length,**ann))
    mapped = pd.DataFrame(mappings).set_index('Geneid')
    reviewed = genes.join(mapped,validate='one_to_one')
    reviewed['exon_union_length_matches_counts'] = reviewed.exon_union_length.eq(reviewed.counted_feature_length)
    reviewed['symbol_differs_between_sources'] = (reviewed.gene_symbol.notna() & reviewed.reference_gene_name.notna() &
        reviewed.gene_symbol.ne(reviewed.reference_gene_name))
    reviewed['annotation_name'] = reviewed.gene_symbol.fillna(reviewed.reference_gene_name)
    reviewed['was_missing_name'] = reviewed.index.isin(queue.index)
    reviewed.to_csv(OUT/'all_1852_annotation_review.tsv',sep='\t')
    reviewed.loc[queue.index].to_csv(OUT/'previously_unnamed_features.tsv',sep='\t')
    annotated_shortlist = reviewed.loc[shortlist.index].copy()
    annotated_shortlist['selection_reason'] = shortlist.selection_reason
    annotated_shortlist.to_csv(OUT/'original_shortlist_annotated.tsv',sep='\t')
    reviewed[reviewed.symbol_differs_between_sources].to_csv(OUT/'names_differing_between_sources.tsv',sep='\t')
    issues = reviewed[reviewed.match_status.ne('exact_version') | ~reviewed.exon_union_length_matches_counts]
    issues.to_csv(OUT/'identifier_or_length_issues.tsv',sep='\t')
    types = reviewed.groupby(['was_missing_name','reference_gene_type'],dropna=False).size().reset_index(name='genes')
    types.to_csv(OUT/'gene_type_summary.tsv',sep='\t',index=False)

    # Explicit companion selection, never overwrite the original 20-gene list.
    eligible = reviewed[reviewed.annotation_name.notna() & reviewed.match_status.eq('exact_version')]
    eligible = eligible.sort_values(['minimum_abs_log2FC_seven','maximum_padj_seven','ensembl_id'],ascending=[False,True,True])
    first = eligible.head(12)
    extra = eligible[eligible.max_abs_marker_partial_r.lt(.3) & ~eligible.index.isin(first.index)].head(8)
    expanded = pd.concat([first.assign(selection_reason='effect_leader'),
                          extra.assign(selection_reason='additional_lower_marker_correlation')])
    assert len(expanded)==20 and expanded.index.is_unique
    expanded['in_original_shortlist'] = expanded.index.isin(shortlist.index)
    expanded.to_csv(OUT/'annotation_expanded_shortlist_20.tsv',sep='\t')
    changed = reviewed.loc[reviewed.index.isin(set(expanded.index)^set(shortlist.index))].copy()
    changed['in_original_shortlist'] = changed.index.isin(shortlist.index)
    changed['in_annotation_expanded_shortlist'] = changed.index.isin(expanded.index)
    changed.to_csv(OUT/'shortlist_membership_changes.tsv',sep='\t')

    # Preserve original reference evidence from the archive without extracting it.
    archive = OUT.parent/'ppmi-eligibility-audit/incoming/PPMI_RNAseq_IR3_Analysis.tar.gz'
    archive_evidence = None
    with tarfile.open(archive,'r|gz') as t:
        for m in t:
            if m.isfile() and '.featureCounts.' in m.name:
                first_line = t.extractfile(m).readline().decode().rstrip()
                assert 'gencode.v29.primary_assembly.annotation.gtf' in first_line
                assert '"-t" "exon"' in first_line and '"-g" "gene_id"' in first_line
                archive_evidence = dict(member=m.name,header=first_line,archive=str(archive),
                    header_sha256=hashlib.sha256(first_line.encode()).hexdigest())
                break
    assert archive_evidence is not None
    (OUT/'original_reference_evidence.json').write_text(json.dumps(archive_evidence,indent=2)+'\n')
    resolved = reviewed.loc[queue.index]
    summary = dict(status='COMPLETE',reference='GENCODE v29 primary assembly',assembly='GRCh38.p12',
        source_url=URL,retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        robust_genes=1852,exact_version_matches=int(reviewed.match_status.eq('exact_version').sum()),
        original_missing_names=525,names_resolved=int(resolved.reference_gene_name.notna().sum()),
        identifier_or_length_issues=len(issues),original_shortlist_genes=20,
        original_shortlist_exact_matches=int(annotated_shortlist.match_status.eq('exact_version').sum()),
        original_shortlist_gene_types=annotated_shortlist.reference_gene_type.value_counts(dropna=False).to_dict(),
        original_shortlist_name_differences=int(annotated_shortlist.symbol_differs_between_sources.sum()),
        expanded_shortlist_added=expanded.loc[~expanded.in_original_shortlist,'annotation_name'].tolist(),
        original_shortlist_retained_in_expanded=int(expanded.in_original_shortlist.sum()),
        no_models_refitted=True,execution='CPU annotation parsing and joins',input_sha256=hashes)
    differences = annotated_shortlist[annotated_shortlist.symbol_differs_between_sources]
    top = resolved.head(12)
    type_text = ', '.join(f'{count} {kind}' for kind,count in summary['original_shortlist_gene_types'].items())
    report = '# Local analysis report\n\nConsult the locally generated tables and diagnostics.\n'
    (OUT/'README.md').write_text(report)
    for path,expected in hashes.items():assert sha256(path)==expected
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    parent = PRIORITY/'README.md';text = parent.read_text()
    link = '../ppmi-gene-annotation-review-2026-09-12/README.md'
    if link not in text:
        parent.write_text(text+f'\n## Reference annotation follow-up\n\n[GENCODE v29 review and annotation-expanded companion shortlist]({link}) resolves {summary["names_resolved"]}/525 missing names. The original shortlist above is preserved.\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':main()
