"""Create styled gene networks and an EnrichmentMap; save a Cytoscape session."""
import json
import math
from pathlib import Path
from urllib.parse import quote

import networkx as nx
import pandas as pd
import requests

OUT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:1234/v1/'
STATE=OUT/'cytoscape_build_state.json'
state=json.loads(STATE.read_text()) if STATE.exists() else {}


def call(method,path,data=None,params=None,timeout=120):
    r=requests.request(method,BASE+path,json=data,params=params,timeout=timeout)
    if not r.ok:raise RuntimeError(f'{method} {path}: HTTP {r.status_code}: {r.text[:3000]}')
    value=r.json() if r.content and 'json' in r.headers.get('Content-Type','') else r.text
    if isinstance(value,dict) and value.get('errors'):raise RuntimeError(value)
    return value


def checkpoint(key,value):
    state[key]=value;STATE.write_text(json.dumps(state,indent=2)+'\n')


def vp(name,value):return {'visualProperty':name,'value':value}


def continuous(column,property,points):
    return dict(mappingType='continuous',mappingColumn=column,mappingColumnType='Double',
                visualProperty=property,points=[dict(value=x,lesser=y,equal=y,greater=y) for x,y in points])


def passthrough(column,property):
    return dict(mappingType='passthrough',mappingColumn=column,mappingColumnType='String',visualProperty=property)


def install_style(style):
    name=style['title'];existing=call('GET','styles')
    if name in existing:call('PUT','styles/'+quote(name,safe=''),style)
    else:call('POST','styles',style)


def view_id(network):
    views=call('GET',f'networks/{network}/views')
    if not views:views=[call('POST',f'networks/{network}/views')]
    return int(views[0])


def export(network,stem,height=1400):
    call('GET',f'apply/fit/{network}')
    r=requests.get(BASE+f'networks/{network}/views/first.png',params={'h':height},timeout=30)
    r.raise_for_status();assert r.content.startswith(b'\x89PNG')
    (OUT/f'{stem}.png').write_bytes(r.content)
    r=requests.get(BASE+f'networks/{network}/views/first.svg',timeout=30)
    r.raise_for_status();assert '<svg' in r.text
    (OUT/f'{stem}.svg').write_text(r.text)
    net=call('GET',f'networks/{network}')
    (OUT/f'{stem}.cyjs').write_text(json.dumps(net,indent=2))
    def scalars(d):return {k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in d.items() if v is not None}
    graph=nx.MultiGraph(**scalars(net.get('data',{})))
    for n in net['elements']['nodes']:graph.add_node(n['data']['id'],**scalars(n['data']))
    for e in net['elements']['edges']:
        d=e['data'];graph.add_edge(d['source'],d['target'],**scalars(d))
    nx.write_graphml(graph,OUT/f'{stem}.graphml')


def main():
    version=call('GET','version');assert version['cytoscapeVersion']=='3.10.4'
    raw=json.loads((OUT/'string_query_response.json').read_text())
    source=int(raw['data']['SUID'])
    genes=pd.read_csv(OUT/'gene_nodes_0.7.tsv',sep='\t')
    columns=['ensembl_id','Geneid','gene_symbol','primary_log2FC','minimum_abs_log2FC_seven',
             'maximum_padj_seven','max_abs_marker_partial_r','most_correlated_immune_score',
             'is_MCP_marker','primary_Cook_flagged','review_flag','STRING_degree','STRING_betweenness_unweighted',
             'STRING_component','STRING_component_size']
    for cutoff in [.4,.7,.9]:
        nodes=pd.read_csv(OUT/f'gene_nodes_{cutoff:.1f}.tsv',sep='\t')[columns].copy()
        nodes['border_color']=nodes.review_flag.map({'none':'#555555','MCP marker':'#762A83','Cook flagged':'#111111'})
        nodes['border_width']=nodes.review_flag.map({'none':1.0,'MCP marker':5.0,'Cook flagged':5.0})
        records=json.loads(nodes.to_json(orient='records'))
        if cutoff==.4:
            network=source
            mapping=pd.read_csv(OUT/'STRING_mapping.tsv',sep='\t').set_index('ensembl_id')
            for record in records:record['SUID']=int(mapping.loc[record['ensembl_id'],'source_SUID'])
            call('PUT',f'networks/{network}/tables/defaultnode',{'key':'SUID','dataKey':'SUID','data':records})
            checkpoint('gene_0.4',network)
        elif f'gene_{cutoff}' not in state:
            edges=pd.read_csv(OUT/f'gene_edges_{cutoff:.1f}.tsv',sep='\t')
            payload={'data':{'name':f'PPMI 20 genes — STRING confidence ≥{cutoff}',
                             'STRING_version':'12','species':'Homo sapiens','confidence':cutoff,
                             'additional_interactors':0,'edge_meaning':'functional association'},
                     'elements':{'nodes':[{'data':dict(id=d['ensembl_id'],name=d['gene_symbol'],**d)} for d in records],
                                 'edges':[{'data':dict(id=f'edge{i}',interaction='STRING association',**d)}
                                          for i,d in enumerate(json.loads(edges.to_json(orient='records')))]}}
            created=call('POST','networks',payload,params={'collection':'PPMI 20-gene network review','format':'json'})
            network=int(created['networkSUID']);checkpoint(f'gene_{cutoff}',network)
        else:network=int(state[f'gene_{cutoff}'])
        if cutoff==.4:continue
        net=call('GET',f'networks/{network}')
        assert len(net['elements']['nodes'])==20
        idmap={n['data']['ensembl_id']:n['data']['SUID'] for n in net['elements']['nodes']}
        view=view_id(network)
        linked=nodes[nodes.STRING_degree.gt(0)].sort_values('STRING_degree',ascending=False)
        isolated=nodes[nodes.STRING_degree.eq(0)].sort_values('gene_symbol')
        positions=[]
        for i,row in enumerate(linked.itertuples()):
            angle=2*math.pi*i/max(len(linked),1)
            positions.append(dict(SUID=int(idmap[row.ensembl_id]),view=[vp('NODE_X_LOCATION',200*math.cos(angle)),vp('NODE_Y_LOCATION',100*math.sin(angle))]))
        for i,row in enumerate(isolated.itertuples()):
            positions.append(dict(SUID=int(idmap[row.ensembl_id]),view=[vp('NODE_X_LOCATION',(i%5-2)*160),vp('NODE_Y_LOCATION',260+(i//5)*135)]))
        call('PUT',f'networks/{network}/views/{view}/nodes',positions)
    style={'title':'PPMI gene evidence','defaults':[vp('NODE_SHAPE','ELLIPSE'),vp('NODE_SIZE',50.0),
        vp('NODE_LABEL_FONT_SIZE',16),vp('NODE_LABEL_COLOR','#222222'),vp('NODE_BORDER_WIDTH',1.0),
        vp('EDGE_WIDTH',3.0),vp('EDGE_STROKE_UNSELECTED_PAINT','#777777'),vp('EDGE_TARGET_ARROW_SHAPE','NONE'),
        vp('NETWORK_BACKGROUND_PAINT','#FFFFFF')],
        'mappings':[passthrough('gene_symbol','NODE_LABEL'),passthrough('border_color','NODE_BORDER_PAINT'),
            continuous('primary_log2FC','NODE_FILL_COLOR',[(-.7,'#2166AC'),(0,'#F7F7F7'),(.7,'#D6604D')]),
            continuous('minimum_abs_log2FC_seven','NODE_SIZE',[(.20,40.0),(.46,85.0)]),
            continuous('border_width','NODE_BORDER_WIDTH',[(1.0,1.0),(5.0,5.0)])]}
    install_style(style)
    for cutoff in [.7,.9]:
        network=int(state[f'gene_{cutoff}'])
        call('GET',f'apply/styles/{quote(style["title"],safe="")}/{network}')
        export(network,f'gene_network_{cutoff}')

    if 'pathway' not in state:
        args={'analysisType':'generic','gmtFile':str(OUT/'selected_pathways_original_definitions.gmt'),
            'enrichmentsDataset1':str(OUT/'enrichmentmap_results.tsv'),
            'ranksDataset1':str(OUT/'primary_ranks.rnk'),'filterByExpressions':'true',
            'pvalue':'1.0','qvalue':'0.05','similaritycutoff':'0.25','coefficients':'JACCARD',
            'networkName':'PPMI robust pathways — shortlist context','runAutoAnnotate':'false'}
        (OUT/'enrichmentmap_command.json').write_text(json.dumps(args,indent=2)+'\n')
        before=set(call('GET','networks'))
        response=call('POST','commands/enrichmentmap/build',args)
        (OUT/'enrichmentmap_response.json').write_text(json.dumps(response,indent=2)+'\n')
        after=set(call('GET','networks'));new=after-before
        assert len(new)==1,(response,new)
        checkpoint('pathway',int(next(iter(new))))
    network=int(state['pathway']);net=call('GET',f'networks/{network}')
    (OUT/'enrichmentmap_raw.cyjs').write_text(json.dumps(net,indent=2))
    print('EnrichmentMap network:',network,'nodes:',len(net['elements']['nodes']),'edges:',len(net['elements']['edges']))
    checkpoint('stage','gene_networks_and_enrichmentmap_created')
    call('POST','commands/session/save',{'file':str(OUT/'PPMI_20_gene_networks.cys')})
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
