import json,re,csv
from pathlib import Path
import pandas as pd

out=Path('outputs/ppmi-model-design-2026-09-12');out.mkdir(exist_ok=True)
manifest=json.loads(Path('outputs/ppmi-expression-qc-2026-09-12/cohort_after_QC.json').read_text())
md=pd.read_csv('outputs/ppmi-expression-qc-2026-09-12/sample_metadata.tsv',sep='\t',dtype=str).set_index('PATNO')
ps=pd.read_csv('outputs/ppmi-extended-source/Participant_Status_11Sep2026.csv',dtype=str).set_index('PATNO')
demo=pd.read_csv('outputs/ppmi-extended-source/Demographics_11Sep2026.csv',dtype=str)
def month(s):
    if isinstance(s,str) and re.fullmatch(r'\d{2}/\d{4}',s):
        m,y=map(int,s.split('/'));return y*12+m
    return None
rows=[];evidence=[]
for r in manifest:
    p=r['PATNO'];m=md.loc[p];v=demo[demo.PATNO==p]
    births=set(v.BIRTHDT.dropna());sexes=set(v.SEX.dropna())
    b=next(iter(births)) if len(births)==1 else None
    clinical_sex={'0':'Female','1':'Male'}.get(next(iter(sexes))) if len(sexes)==1 else None
    c=r['collection_months'][0];cm=month(c);bm=month(b)
    age=(cm-bm)/12 if cm is not None and bm is not None else None
    enrollment=ps.loc[p] if p in ps.index else None
    rows.append(dict(PATNO=p,group=r['analysis_group'],collection_month=c,age_collection_years=age,sex=clinical_sex,expression_sex=m.GENDER,sex_consistent=clinical_sex==m.GENDER,phase=m.Sample.split('.')[0].replace('-IR1',''),plate=m.Plate,RIN=m['RIN Value'],intergenic_percent=m.PCT_INTERGENIC_BASES,usable_percent=m.PCT_USABLE_BASES,mapping_percent=m.uniquely_mapped_percent,coverage_bias=m.MEDIAN_5PRIME_TO_3PRIME_BIAS,medication_timing_sensitivity_exclude=r['exclude_in_medication_timing_sensitivity'],enrollment_age=enrollment.ENROLL_AGE if enrollment is not None else None))
    evidence.append(dict(PATNO=p,demographics_csv_rows=[int(i)+2 for i in v.index],birth_month=b,clinical_sex_values=sorted(sexes),collection_month=c,age_method='difference of released collection and birth year-month divided by 12; approximate age, not exact day age'))
d=pd.DataFrame(rows)
for c in ['RIN','intergenic_percent','usable_percent','mapping_percent','coverage_bias','enrollment_age']:d[c]=pd.to_numeric(d[c],errors='coerce')
d['batch']=d.phase+'__plate_'+d.plate
assert len(d)==558 and d.PATNO.is_unique
d.to_csv(out/'covariates.tsv',sep='\t',index=False)
(out/'covariate_provenance.json').write_text(json.dumps(evidence,indent=2))
print('Missing values:',d.isna().sum().to_dict());print('Sex mismatches:',d.loc[~d.sex_consistent,['PATNO','sex','expression_sex']].to_dict('records'))
print(d.groupby('group')[['age_collection_years','RIN','intergenic_percent','usable_percent']].agg(['mean','std','min','max']).round(2).to_string())
print(pd.crosstab(d.batch,d.group).to_string());print(pd.crosstab(d.sex,d.group).to_string())
