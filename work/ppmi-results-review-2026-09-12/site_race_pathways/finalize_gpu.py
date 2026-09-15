"""Validate enrichment on CUDA and compare pathways with prior results."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
REVIEW = OUT.parent
ANALYSIS = REVIEW.parent / 'ppmi-deseq2-2026-09-12'
sys.path.insert(0, str(ANALYSIS))
from gpu_postprocess import bh


def markdown(frame):
    def value(x):
        if pd.isna(x):
            return '—'
        return f'{x:.4g}' if isinstance(x, float) else str(x).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(value(x) for x in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def main():
    assert torch.cuda.is_available(), 'CUDA is required for validation'
    devices = [i for i in range(torch.cuda.device_count())
               if 'RTX A3000' in torch.cuda.get_device_name(i)]
    assert devices, 'The requested NVIDIA RTX A3000 is not available'
    torch.cuda.set_device(devices[0])
    tensor = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), device='cuda', dtype=torch.float64)
    summary = json.loads((OUT / 'enrichment_summary.json').read_text())
    assert summary['status'] == 'COMPLETE'
    hashes = {}
    for row in pd.read_csv(OUT / 'input_md5.tsv', sep='\t').itertuples():
        data = Path(row.path).read_bytes()
        assert hashlib.md5(data).hexdigest() == row.md5
        hashes[row.path] = hashlib.sha256(data).hexdigest()
    sources = {'Hallmark': ANALYSIS / 'primary/hallmark_enrichment.tsv',
               'C2': ANALYSIS / 'c2_c5bp/C2_enrichment.tsv',
               'C5_BP': ANALYSIS / 'c2_c5bp/C5_BP_enrichment.tsv'}
    metadata = pd.read_csv(OUT / 'pathway_metadata.tsv', sep='\t').rename(columns={'gs_name': 'pathway'})
    comparisons, records, validation = [], [], {}
    for collection, source in sources.items():
        prior = pd.read_csv(source, sep='\t')
        current = pd.read_csv(OUT / f'{collection}_enrichment.tsv', sep='\t')
        assert current.pathway.is_unique and prior.pathway.is_unique
        observed = bh(tensor(current.pval))
        torch.testing.assert_close(observed, tensor(current.padj), rtol=1e-8, atol=1e-12)
        validation[collection] = float((observed-tensor(current.padj)).abs().max())
        compare = prior[['pathway', 'NES', 'padj', 'leadingEdge']].merge(
            current[['pathway', 'NES', 'padj', 'leadingEdge']], on='pathway', how='outer',
            suffixes=('_primary', '_site_race'), validate='one_to_one')
        compare['collection'] = collection
        compare['tested_both'] = compare.NES_primary.notna() & compare.NES_site_race.notna()
        compare['same_direction'] = compare.tested_both & (np.sign(compare.NES_primary) == np.sign(compare.NES_site_race))
        compare['significant_same_direction_both'] = compare.same_direction & compare.padj_primary.lt(.05) & compare.padj_site_race.lt(.05)
        def overlap(row):
            if not row.tested_both:
                return np.nan
            a, b = set(str(row.leadingEdge_primary).split(';')), set(str(row.leadingEdge_site_race).split(';'))
            return len(a & b) / len(a | b)
        compare['leading_edge_jaccard'] = compare.apply(overlap, axis=1)
        compare = compare.merge(metadata[metadata.collection == collection].drop(columns='collection'),
                                on='pathway', how='left', validate='one_to_one')
        comparisons.append(compare)
        records.append(dict(collection=collection,site_race_tested=len(current),
                            primary_significant=int(prior.padj.lt(.05).sum()),
                            site_race_significant=int(current.padj.lt(.05).sum()),
                            retained_same_direction=int(compare.significant_same_direction_both.sum()),
                            significant_direction_reversals=int((compare.padj_primary.lt(.05)&compare.padj_site_race.lt(.05)&~compare.same_direction).sum())))
    comparison = pd.concat(comparisons, ignore_index=True)
    # Match the optional joint testing family used in the primary C2/C5_BP extension.
    joint = comparison.collection.isin(['C2', 'C5_BP']) & comparison.padj_site_race.notna()
    pvalues = pd.concat([pd.read_csv(OUT / f'{c}_enrichment.tsv', sep='\t').assign(collection=c)
                        for c in ['C2', 'C5_BP']], ignore_index=True).set_index(['collection', 'pathway'])
    joint_index = pd.MultiIndex.from_frame(comparison.loc[joint, ['collection', 'pathway']])
    comparison.loc[joint, 'padj_joint_C2_C5_BP_site_race'] = bh(tensor(pvalues.loc[joint_index, 'pval'])).cpu().numpy()
    comparison.to_csv(OUT / 'pathway_comparison.tsv', sep='\t', index=False)
    comparison[comparison.significant_same_direction_both].to_csv(OUT / 'retained_pathways.tsv', sep='\t', index=False)
    hallmark = pd.read_csv(ANALYSIS / 'hallmark_sensitivity_comparison.tsv', sep='\t')
    hallmark = hallmark.merge(comparison[comparison.collection == 'Hallmark'][['pathway', 'NES_site_race', 'padj_site_race']],
                               on='pathway', how='left', validate='one_to_one')
    hallmark['significant_same_direction_all_six'] = (hallmark.significant_same_direction_all_models &
        hallmark.padj_site_race.lt(.05) & (np.sign(hallmark.primary_NES) == np.sign(hallmark.NES_site_race)))
    hallmark.to_csv(OUT / 'hallmark_six_model_comparison.tsv', sep='\t', index=False)
    original = hallmark[hallmark.significant_same_direction_all_models]
    retained = int(original.significant_same_direction_all_six.sum())
    table = pd.DataFrame(records)
    table.to_csv(OUT / 'summary.tsv', sep='\t', index=False)
    report = '# Local analysis report\n\nConsult the locally generated tables and diagnostics.\n'
    (OUT / 'README.md').write_text(report)
    parent = REVIEW / 'README.md'
    text = parent.read_text()
    link = 'site_race_pathways/README.md'
    if link not in text:
        parent.write_text(text + f'\n## Completed pathway extension\n\nSite/race-adjusted enrichment is now complete; {retained}/15 original robust Hallmark pathways retain significance and direction across six models. See the [pathway sensitivity report]({link}). The preceding review describes the earlier analysis stage.\n')
    (OUT / 'validation.json').write_text(json.dumps(dict(status='COMPLETE',
        gpu=torch.cuda.get_device_name(torch.cuda.current_device()),BH_max_abs_errors=validation,input_sha256=hashes,
        original_robust_hallmark_retained_all_six=retained,collections=records), indent=2)+'\n')
    print(table.to_string(index=False))
    print(f'Complete: {retained}/15 original Hallmark pathways retained across six models.')


if __name__ == '__main__':
    main()
