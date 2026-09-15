import os
import csv,json,re,io,zipfile,calendar
from pathlib import Path
from datetime import date
from collections import defaultdict,Counter

src=Path('outputs/ppmi-extended-source'); out=Path('outputs/ppmi-ir3-eligibility-audit');out.mkdir(exist_ok=True)
tables={}; grouped={}
def add(name,txt):
    rows=list(csv.DictReader(io.StringIO(txt))); tables[name]=rows; grouped[name]=defaultdict(list)
    for i,r in enumerate(rows,2):
        r['_source']=name+'_11Sep2026.csv';r['_row']=i
        grouped[name][r.get('PATNO','')].append(r)
for f in src.glob('*.csv'):
    if f.name.startswith(('Data_Dictionary','Code_List','Deprecated','ST_CATALOG')):continue
    add(f.name.removesuffix('_11Sep2026.csv'),f.read_text(encoding='utf-8-sig'))
with zipfile.ZipFile(os.environ['PPMI_MEDICAL_ARCHIVE']) as z:
    for name in z.namelist():
        if name.endswith('.csv'):add(name.removesuffix('_11Sep2026.csv'),z.read(name).decode('utf-8-sig'))
def get(name,p):return grouped[name].get(p,[])
def interval(s):
    if re.fullmatch(r'\d{2}/\d{4}',s or ''):
        m,y=map(int,s.split('/'));return (date(y,m,1),date(y,m,calendar.monthrange(y,m)[1]))
    if re.fullmatch(r'\d{4}',s or ''):return date(int(s),1,1),date(int(s),12,31)
    return None
def timing(s,cut):
    x=interval(s)
    if not x or not cut:return 'unknown'
    return 'before' if x[1]<cut[0] else 'after' if x[0]>cut[1] else 'overlap'
def ref(r,fields):return dict(source=r['_source'],csv_row=r['_row'],REC_ID=r.get('REC_ID'),**{k:r.get(k,'') for k in fields})
def norm(s):return str(int(s)) if s.isdigit() else s
meta=list(csv.DictReader(Path('outputs/ppmi-ir3-archive-check/metaDataIR3.csv').open()))
members=[json.loads(x) for x in Path('outputs/ppmi-ir3-archive-check/members.jsonl').read_text().splitlines()]
countmap=defaultdict(list)
for m in members:
    if m['file'] and '.featureCounts.' in m['name']:
        parts=Path(m['name']).name.split('.')
        countmap[tuple(parts[1:5])].append(m['name'])
results=[]
for m in meta:
    pid=m['PATNO'];visit=m['CLINICAL_EVENT'];pool=m['DIAGNOSIS'] in ('PDPOOL','HCPOOL')
    key=(pid,visit,m['Specimen Bar Code'],m['HudAlphaID'])
    # Pooled controls use POOL in file names and Unk in metadata.
    files=countmap.get(key,[]) if not pool else countmap.get((pid,'POOL',key[2],key[3]),[])
    file_match_basis='metadata_identity_fields'
    if not files:
        # Match the complete sample identity explicitly supplied by PPMI, preserving any R suffix.
        sample_key=tuple(m['Sample'].split('.')[1:5])
        if len(sample_key)==4 and sample_key[1:]==key[1:]:
            files=countmap.get(sample_key,[])
            file_match_basis='PPMI_sample_string_with_documented_PATNO_difference'
    labs=[r for name in ('Laboratory_Procedures-Archived','Research_Biospecimens') for r in get(name,pid) if r['EVENT_ID']==visit]
    dates=sorted({r['BLDDRDT'] for r in labs if r.get('BLDDRDT')})
    cut=interval(dates[0]) if len(dates)==1 else None
    match='linked_at_participant_visit_month' if cut and len(files)==1 else 'unresolved'
    dx=[r for name in ('Primary_Diagnosis-Archived','Primary_Research_Diagnosis') for r in get(name,pid) if r['EVENT_ID']==visit]
    dx_basis='same_visit'
    if not dx and visit=='BL':
        dx=[r for name in ('Primary_Diagnosis-Archived','Primary_Research_Diagnosis') for r in get(name,pid) if r['EVENT_ID']=='SC' and timing(r.get('INFODT'),cut) in ('before','overlap')]
        dx_basis='screening_before_or_same_month_as_baseline'
    codes={norm(r['PRIMDIAG']) for r in dx if r.get('PRIMDIAG')}
    diagnosis='PD_supported' if codes=={'1'} else 'no_neurological_disorder_supported' if codes=={'17'} else 'other_or_conflicting_diagnosis' if codes else 'unresolved'
    treatment=[];possible=[]
    # The laboratory item refers to PD medication at specimen collection.
    for r in labs:
        if r.get('PDMEDYN')=='1':treatment.append(dict(kind='PD_medication_at_collection',**ref(r,['EVENT_ID','BLDDRDT','PDMEDYN'])))
    for r in get('Use_of_PD_Medication-Archived',pid):
        when=timing(r.get('INFODT'),cut)
        if r.get('PDMEDYN')=='1' and when in ('before','overlap'):
            item=dict(kind='PD_medication_visit_report',timing=when,**ref(r,['EVENT_ID','INFODT','PDMEDYN']))
            (treatment if when=='before' else possible).append(item)
    for r in get('Concomitant_Medications-Archived',pid):
        if r.get('DISMED')!='1' and r.get('PD_MOTOR_MED')!='1':continue
        when=timing(r.get('STARTDT'),cut)
        if when=='after':continue
        indication=r.get('CMINDC','').strip()
        direct=bool(re.search(r'parkinson|^pd$|^pd\s',indication,re.I)) and not re.search(r'not|non.parkinson',indication,re.I)
        item=dict(kind='archived_medication',timing=when,explicit_PD_indication=direct,**ref(r,['CMTRT','CMINDC','STARTDT','STOPDT','DISMED','PD_MOTOR_MED']))
        (treatment if direct and when=='before' else possible).append(item)
    for name in ('Procedure_for_PD_Log','Surgery_for_Parkinson_Disease-Archived'):
        for r in get(name,pid):
            if name.endswith('Archived') and r.get('PDSURG')!='1':continue
            when=timing(r.get('PDSURGDT'),cut)
            if when=='after':continue
            item=dict(kind='PD_procedure',timing=when,**ref(r,['PDSURGDT','PDSURGTP','PDSRGTPC']))
            (treatment if when=='before' else possible).append(item)
    for r in get('Initiation_of_Dopaminergic_Therapy',pid):
        if r.get('DOPTHERST')=='1' and timing(r.get('INFODT'),cut)!='after':
            # Does not independently establish indication or exact initiation date.
            possible.append(dict(kind='dopaminergic_initiation_report',**ref(r,['INFODT','EVENT_ID','DOPTHERST'])))
    med_status='prior_or_collection_PD_treatment_documented' if treatment else 'timing_or_indication_needs_review' if possible else 'no_positive_record_found_lifetime_negative_unproven'
    health=[]
    for r in get('General_Medical_History-Archived',pid):
        if r.get('MHHX')=='1' and r.get('MHACTRES')=='1' and timing(r.get('INFODT'),cut) in ('before','overlap'):
            health.append(dict(kind='active_at_history_assessment_not_necessarily_collection',**ref(r,['EVENT_ID','INFODT','MHTERM','MHACTRES','MHDIAGYR'])))
    for r in get('Current_Medical_Conditions_Log-Archived',pid):
        if timing(r.get('DIAGYR'),cut) not in ('before','overlap'):continue
        if r.get('RESOLVD')=='1' and timing(r.get('RESYR'),cut)=='before':continue
        health.append(dict(kind='condition_not_known_resolved_before_collection',**ref(r,['CONDTERM','DIAGYR','RESOLVD','RESYR'])))
    for r in get('Medical_Conditions_Log',pid):
        onset=r.get('MHDIAGDT') or r.get('MHDIAGYR')
        if timing(onset,cut) not in ('before','overlap'):continue
        resolution=r.get('RESDT') or r.get('RESYR')
        if timing(resolution,cut)=='before':continue
        health.append(dict(kind='harmonized_condition_review',**ref(r,['INFODT','MHTERM','MHDIAGDT','MHDIAGYR','RESOLVD','RESDT','RESYR'])))
    if pool:decision='EXCLUDED_POOL'
    elif m['QCflagIR3']!='pass':decision='EXCLUDED_RELEASE_QC'
    elif match=='unresolved':decision='UNRESOLVED_SAMPLE_LINK'
    elif diagnosis=='PD_supported' and treatment:decision='EXCLUDED_PD_TREATMENT'
    elif diagnosis=='PD_supported':decision='UNRESOLVED_PD_LIFETIME_TREATMENT'
    elif m['DIAGNOSIS']=='Control' and codes and '17' not in codes:decision='EXCLUDED_CONTROL_NEUROLOGICAL_DIAGNOSIS'
    elif diagnosis=='no_neurological_disorder_supported':decision='UNRESOLVED_CONTROL_HEALTH'
    else:decision='UNRESOLVED_DIAGNOSIS'
    results.append(dict(PATNO=pid,visit=visit,metadata_label=m['DIAGNOSIS'],sample=m['Sample'],specimen=m['Specimen Bar Code'],facility_id=m['HudAlphaID'],count_files=files,file_match_basis=file_match_basis,release_QC=m['QCflagIR3'],RIN=m['RIN Value'],total_reads=m['total_reads'],uniquely_mapped_percent=m['uniquely_mapped_percent'],sample_link=match,collection_months=dates,diagnosis=diagnosis,diagnosis_basis=dx_basis,diagnosis_codes=sorted(codes),diagnosis_evidence=[ref(r,['EVENT_ID','INFODT','PRIMDIAG','OTHNEURO']) for r in dx],collection_evidence=[ref(r,['EVENT_ID','INFODT','BLDDRDT','BLDRNA','PDMEDYN']) for r in labs],treatment_status=med_status,treatment_evidence=treatment,treatment_review=possible,health_review=health,decision=decision,independent_expression_QC='pending'))
(out/'sample_evidence.json').write_text(json.dumps(results,indent=2))
summary={}
for title,rows in [('all',results),('baseline',[r for r in results if r['visit']=='BL'])]:
    summary[title]={'n':len(rows),'decisions':dict(Counter(r['decision'] for r in rows)),'diagnosis':dict(Counter(r['diagnosis'] for r in rows)),'single_count_file':sum(len(r['count_files'])==1 for r in rows),'clinical_month_link':sum(r['sample_link']!='unresolved' for r in rows),'verified_eligible':0}
summary['baseline_release_pass_by_metadata_label']={}
for label in sorted({r['metadata_label'] for r in results if r['visit']=='BL'}):
    rows=[r for r in results if r['visit']=='BL' and r['release_QC']=='pass' and r['metadata_label']==label]
    summary['baseline_release_pass_by_metadata_label'][label]={'n':len(rows),'decisions':dict(Counter(r['decision'] for r in rows))}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
assert len(results)==4871
assert sum(summary['all']['decisions'].values())==4871
