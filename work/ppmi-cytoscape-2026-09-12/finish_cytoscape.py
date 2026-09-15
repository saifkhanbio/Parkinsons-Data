"""Validate native EnrichmentMap edges, annotate effects, and export the session."""
import json
import math
import textwrap
import zipfile
from urllib.parse import quote

import networkx as nx
import pandas as pd

from build_cytoscape import OUT, call, checkpoint, continuous, export, install_style, passthrough, state, view_id, vp


def main():
    network = int(state['pathway'])
    net = call('GET', f'networks/{network}')
    paths = pd.read_csv(OUT / 'pathway_nodes.tsv', sep='\t').set_index('pathway')
    edges = pd.read_csv(OUT / 'pathway_overlap_edges.tsv', sep='\t')
    nodes = {n['data']['name']: n['data'] for n in net['elements']['nodes']}
    assert set(nodes) == set(paths.index)
    names = {d['id']: name for name, d in nodes.items()}
    expected = {frozenset((r.source, r.target)): r for r in edges.itertuples()}
    actual = set()
    edge_records = []
    for item in net['elements']['edges']:
        d = item['data']
        pair = frozenset((names[d['source']], names[d['target']]))
        assert pair not in actual
        actual.add(pair)
        ref = expected[pair]
        assert abs(d['EnrichmentMap_similarity_coefficient'] - ref.jaccard) < 1e-12
        assert d['EnrichmentMap_Overlap_size'] == ref.shared_genes
        edge_records.append(dict(SUID=int(d['SUID']), jaccard=ref.jaccard, shared_genes=ref.shared_genes))
    assert actual == set(expected)
    call('PUT', f'networks/{network}/tables/defaultedge', {'key': 'SUID', 'dataKey': 'SUID', 'data': edge_records})
    records = []
    for name, row in paths.iterrows():
        d = nodes[name]
        assert d['EnrichmentMap_gs_size'] == row.measured_genes
        assert abs(d['EnrichmentMap_fdr_qvalue_Dataset_1_'] - row.padj_primary) < 1e-12
        record = json.loads(row.to_json(double_precision=15))
        record.update(SUID=int(d['SUID']), pathway=name,
                      pathway_size=float(row.measured_genes),
                      label_wrapped=textwrap.fill(row.display_label.title(), width=26),
                      collection_border={'Hallmark': '#1B9E77', 'C2': '#7570B3', 'C5_BP': '#D95F02'}[row.collection])
        # Generic import uses signed phenotype as a placeholder NES. Restore actual GSEA NES.
        record['EnrichmentMap::NES (Dataset 1)'] = float(row.NES_primary)
        records.append(record)
    call('PUT', f'networks/{network}/tables/defaultnode', {'key': 'SUID', 'dataKey': 'SUID', 'data': records})
    style = {'title': 'PPMI pathway evidence', 'defaults': [
        vp('NODE_SHAPE', 'ELLIPSE'), vp('NODE_SIZE', 65.0), vp('NODE_LABEL_FONT_SIZE', 11),
        vp('NODE_LABEL_COLOR', '#222222'), vp('NODE_BORDER_WIDTH', 3.0),
        vp('EDGE_STROKE_UNSELECTED_PAINT', '#AAAAAA'), vp('EDGE_TARGET_ARROW_SHAPE', 'NONE'),
        vp('NETWORK_BACKGROUND_PAINT', '#FFFFFF')], 'mappings': [
        passthrough('label_wrapped', 'NODE_LABEL'), passthrough('collection_border', 'NODE_BORDER_PAINT'),
        continuous('NES_primary', 'NODE_FILL_COLOR', [(-3.0, '#2166AC'), (0.0, '#F7F7F7'), (3.0, '#D6604D')]),
        continuous('pathway_size', 'NODE_SIZE', [(15.0, 45.0), (500.0, 95.0)]),
        continuous('jaccard', 'EDGE_WIDTH', [(.25, 1.0), (1.0, 5.0)])]}
    install_style(style)
    call('GET', f'apply/styles/{quote(style["title"], safe="")}/{network}')
    graph = nx.Graph()
    graph.add_nodes_from(sorted(paths.index))
    graph.add_edges_from(edges[['source', 'target']].itertuples(index=False, name=None))
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), sorted(c)))
    positions = []
    component_rows = []
    x_cursor = y_cursor = row_height = 0.0
    for index, component in enumerate(components, 1):
        members = sorted(component)
        if len(members) == 1:
            layout = {members[0]: (0.0, 0.0)}
            width, height = 220.0, 150.0
        else:
            layout = nx.kamada_kawai_layout(graph.subgraph(members), weight=None)
            # Scale to keep long pathway labels separated in the saved image.
            closest = min(math.dist(layout[a], layout[b]) for i, a in enumerate(members) for b in members[i+1:])
            scale = 215.0 / closest
            min_x = min(p[0] for p in layout.values())
            min_y = min(p[1] for p in layout.values())
            layout = {k: ((v[0]-min_x)*scale, (v[1]-min_y)*scale) for k, v in layout.items()}
            width = max(p[0] for p in layout.values()) + 220.0
            height = max(p[1] for p in layout.values()) + 150.0
        if x_cursor and x_cursor + width > 1900:
            x_cursor = 0.0
            y_cursor += row_height + 60.0
            row_height = 0.0
        for name, (x, y) in layout.items():
            positions.append(dict(SUID=int(nodes[name]['SUID']), view=[
                vp('NODE_X_LOCATION', float(x+x_cursor+110)), vp('NODE_Y_LOCATION', float(y+y_cursor+75))]))
            component_rows.append(dict(pathway=name, component=index, component_size=len(members)))
        x_cursor += width + 60.0
        row_height = max(row_height, height)
    call('PUT', f'networks/{network}/views/{view_id(network)}/nodes', positions)
    pd.DataFrame(component_rows).to_csv(OUT / 'pathway_components.tsv', sep='\t', index=False)
    call('PUT', 'networks/views/currentNetworkView', {'networkViewSUID': view_id(network)})
    export(network, 'pathway_enrichmentmap', height=3600)
    saved = json.loads((OUT / 'pathway_enrichmentmap.cyjs').read_text())
    for n in saved['elements']['nodes']:
        d = n['data']
        assert abs(d['NES_primary'] - paths.loc[d['pathway'], 'NES_primary']) < 1e-12
        assert abs(d['EnrichmentMap_NES_Dataset_1_'] - d['NES_primary']) < 1e-12
    call('PUT', 'networks/views/currentNetworkView', {'networkViewSUID': view_id(int(state['gene_0.7']))})
    call('POST', 'commands/session/save', {'file': str(OUT / 'PPMI_20_gene_networks.cys')})
    with zipfile.ZipFile(OUT / 'PPMI_20_gene_networks.cys') as archive:
        assert archive.testzip() is None and len(archive.namelist()) > 0
    summary = json.loads((OUT / 'analysis_summary.json').read_text())
    summary.update(status='COMPLETE', native_EnrichmentMap_validation='All 51 sizes, 51 FDRs, 50 edge pairs, shared counts and Jaccards validated; actual NES restored after generic import',
                   pathway_components=len(components), pathway_isolates=len(list(nx.isolates(graph))),
                   cytoscape_version=call('GET', 'version'), session='PPMI_20_gene_networks.cys')
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    checkpoint('stage', 'complete')
    print(json.dumps({k: summary[k] for k in ['status', 'pathway_components', 'pathway_isolates', 'native_EnrichmentMap_validation']}, indent=2))


if __name__ == '__main__':
    main()
