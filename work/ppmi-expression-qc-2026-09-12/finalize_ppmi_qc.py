import json
from pathlib import Path
import pandas as pd

out=Path('outputs/ppmi-expression-qc-2026-09-12')
q=pd.read_csv(out/'sample_QC.tsv',sep='\t',dtype={'PATNO':str})
m=pd.read_csv(out/'sample_metadata.tsv',sep='\t',dtype={'PATNO':str})
d=q.merge(m[['PATNO','PCT_USABLE_BASES','PCT_INTERGENIC_BASES']],on='PATNO',validate='one_to_one')
thresholds={}
for col,side in [('PCT_USABLE_BASES','low'),('PCT_INTERGENIC_BASES','high')]:
    x=pd.to_numeric(d[col],errors='coerce');median=x.median();scale=1.4826*(x-median).abs().median()
    cut=median+(-3 if side=='low' else 3)*scale
    d[side+'_'+col]=(x<cut) if side=='low' else (x>cut)
    thresholds[col]={'median':float(median),'scaled_MAD':float(scale),'cutoff':float(cut),'tail':side}
d['excluded_independent_QC']=(d.low_library|d.low_detection|d.extreme_PC)&d.low_PCT_USABLE_BASES&d.high_PCT_INTERGENIC_BASES
d['disposition']=d.excluded_independent_QC.map({True:'EXCLUDE_CORROBORATED_TECHNICAL_OUTLIER',False:'RETAIN_WITH_QC_COVARIATES_AND_FLAGS'})
d['reason']=d.excluded_independent_QC.map({True:'Expression/library outlier plus low usable bases and high intergenic bases (>3 scaled MAD in adverse tails)',False:'No joint technical exclusion; individual screening flags preserved'})
d.to_csv(out/'QC_disposition.tsv',sep='\t',index=False)
manifest=json.loads(Path('outputs/ppmi-broader-cohort-2026-09-12/cohort_manifest.json').read_text())
lookup=d.set_index('PATNO').to_dict('index')
for r in manifest:
    z=lookup[r['PATNO']]
    r['independent_expression_QC']=z['disposition'];r['QC_reason']=z['reason']
    r['cohort_status']='EXCLUDED_INDEPENDENT_QC' if z['excluded_independent_QC'] else 'INCLUDED_BROADER_COHORT_AFTER_QC'
    r['QC_screening_flag']=bool(z['review_flag'])
kept=[r for r in manifest if r['cohort_status']=='INCLUDED_BROADER_COHORT_AFTER_QC']
(out/'cohort_after_QC.json').write_text(json.dumps(kept,indent=2))
(out/'all_sample_dispositions.json').write_text(json.dumps(manifest,indent=2))
counts=lambda rs:{g:sum(r['analysis_group']==g for r in rs) for g in ['PD','Control']}
summary={'status':'QC_COMPLETE','initial':counts(manifest),'retained':counts(kept),'excluded':counts([r for r in manifest if r not in kept]),'retained_total':len(kept),'medication_timing_sensitivity':counts([r for r in kept if not r['exclude_in_medication_timing_sensitivity']]),'retained_with_individual_review_flags':sum(r['QC_screening_flag'] for r in kept),'thresholds':thresholds,'rule_provenance':'Corroborating technical rule operationalized after first QC inspection, before differential expression or classifier performance evaluation; not a universal published cutoff and not fully prespecified. Evaluate sensitivity to QC choices.','excluded_PATNO':d.loc[d.excluded_independent_QC,'PATNO'].tolist()}
(out/'final_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
