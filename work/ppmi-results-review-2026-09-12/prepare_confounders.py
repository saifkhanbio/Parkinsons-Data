"""Audit archived covariates; use only explicit fields and matched source rows."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
WORK=OUT.parent
ANALYSIS=WORK/'ppmi-deseq2-2026-09-12'


def main():
    d=pd.read_csv(ANALYSIS/'primary_colData.tsv',sep='\t',dtype={'PATNO':str})
    old=pd.read_csv(WORK/'ppmi-model-design-2026-09-12/DESeq2_colData.tsv',sep='\t',dtype={'PATNO':str})
    months=old.set_index('PATNO').collection_month
    archive=WORK/'ppmi-eligibility-audit/incoming/download.zip'
    inventory=[]
    with zipfile.ZipFile(archive) as z:
        demo=pd.read_csv(z.open('Demographics_11Sep2026.csv'),dtype=str).set_index('PATNO')
        lab=pd.read_csv(z.open('Laboratory_Procedures_with_Elapsed_Times_11Sep2026.csv'),dtype=str)
        code=pd.read_csv(z.open('Code_List_-_Harmonized_11Sep2026.csv'),dtype=str)
    for file in (archive,WORK/'ppmi-eligibility-audit/incoming/Medical.zip'):
        with zipfile.ZipFile(file) as z:
            for name in z.namelist():
                if name.endswith('.csv'):
                    columns=pd.read_csv(z.open(name),nrows=0).columns.tolist()
                    inventory.append(dict(archive=file.name,member=name,columns=columns))
    evidence=[]
    extra=[]
    race_fields=['RAWHITE','RAASIAN','RABLACK','RAHAWOPI','RAINDALS','RANOS']
    for p in d.PATNO:
        row=demo.loc[p]
        assert isinstance(row,pd.Series) and row.PAG_NAME=='SCREEN'
        # Categories describe checked self-report fields, not genetic ancestry.
        codes=[row[f] for f in race_fields]
        unknown=row.RAUNKNOWN=='1' or any(v not in ('0','1') for v in codes)
        race='unknown_or_unreported' if unknown or not any(v=='1' for v in codes) else (
            'white_only' if row.RAWHITE=='1' and not any(row[f]=='1' for f in race_fields[1:]) else 'other_or_multiple_reported')
        candidates=lab[(lab.PATNO==p)&(lab.EVENT_ID=='BL')&(lab.BLDDRDT==months[p])]
        sites=candidates.CNO.dropna().unique()
        fasting=candidates.FASTSTAT.dropna().unique()
        clocks=candidates.BLDRNATM.dropna().unique()
        extra.append(dict(PATNO=p,site=sites[0] if len(sites)==1 else None,
                          race_category=race,HISPLAT=row.HISPLAT,ASHKJEW=row.ASHKJEW,
                          fasting_status=fasting[0] if len(fasting)==1 else None,
                          RNA_collection_clock=clocks[0] if len(clocks)==1 else None,
                          matched_lab_rows=len(candidates)))
        evidence.append(dict(PATNO=p,demographic_REC_ID=row.REC_ID,
                             lab_csv_rows=[int(i)+2 for i in candidates.index],
                             lab_REC_IDs=candidates.REC_ID.tolist(),
                             match_rule='PATNO, EVENT_ID=BL, BLDDRDT=RNA collection month',
                             race_field_codes={f:row[f] for f in race_fields+['RAUNKNOWN']}))
    extra=pd.DataFrame(extra)
    d=d.merge(extra,on='PATNO',validate='one_to_one',sort=False)
    d.to_csv(OUT/'confounder_colData.tsv',sep='\t',index=False)
    (OUT/'covariate_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (OUT/'archive_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    relevant=code[(code.MOD_NAME=='SCREEN')&code.ITM_NAME.isin(race_fields+['RAUNKNOWN','HISPLAT','ASHKJEW']) |
                  ((code.MOD_NAME=='LAB')&code.ITM_NAME.isin(['FASTSTAT']))]
    relevant.to_csv(OUT/'used_code_lists.tsv',sep='\t',index=False)
    # Symmetric screening-history extraction avoids the control-focused prior adjudication.
    with zipfile.ZipFile(archive) as z:
        history=pd.read_csv(z.open('General_Medical_History-Archived_11Sep2026.csv'),dtype=str)
    history=history[history.PATNO.isin(d.PATNO)&history.EVENT_ID.isin(['SC','BL'])].copy()
    history['source_csv_row']=history.index+2
    history['collection_month']=history.PATNO.map(months)
    def month(s):
        if isinstance(s,str) and re.fullmatch(r'\d{2}/\d{4}',s):
            m,y=map(int,s.split('/'));return 12*y+m
        return np.nan
    history['assessment_month']=history.INFODT.map(month)
    history['collection_month_index']=history.collection_month.map(month)
    history=history[history.assessment_month.notna() & (history.assessment_month<=history.collection_month_index)]
    text=history.MHTERM.fillna('')+' '+history.VERBATIM.fillna('')
    patterns={'diabetes_mention':r'\bdiabet', 'hypertension_mention':r'\bhypertension\b|high blood pressure',
              'lipid_disorder_mention':r'hyperlipid|hypercholesterol|high cholesterol',
              'allergy_asthma_mention':r'allerg|\basthma\b',
              'smoking_mention':r'\bsmok|\btobacco\b'}
    for label,pattern in patterns.items():
        history[label]=text.str.contains(pattern,case=False,regex=True)
        present=set(history.loc[history[label],'PATNO'])
        d[label]=d.PATNO.isin(present)
    d['history_rows_before_collection']=d.PATNO.map(history.groupby('PATNO').size()).fillna(0).astype(int)
    history[['PATNO','source_csv_row','REC_ID','EVENT_ID','INFODT','MHTERM','VERBATIM','MHACTRES']+list(patterns)].to_csv(
        OUT/'history_screening_evidence.tsv',sep='\t',index=False)
    d.to_csv(OUT/'confounder_colData.tsv',sep='\t',index=False)
    summary={'samples':len(d),'site_missing':int(d.site.isna().sum()),'sites':int(d.site.nunique()),
             'race_by_group':pd.crosstab(d.race_category,d.group).to_dict(),
             'fasting_missing':int(d.fasting_status.isna().sum()),
             'RNA_collection_clock_missing':int(d.RNA_collection_clock.isna().sum()),
             'history_screening':'regex mentions in SC/BL histories assessed no later than collection month; non-mentions are not clinical negatives',
             'measured_CBC_differentials_available':False,'smoking_exposure_measurements_available':False,
             'genetic_ancestry_PCs_available':False}
    (OUT/'confounder_availability.json').write_text(json.dumps(summary,indent=2)+'\n')
    hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [archive,ANALYSIS/'primary/results.tsv',OUT/'MCPcounter_genes.tsv',OUT/'REVIEW_PLAN.md']}
    (OUT/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
