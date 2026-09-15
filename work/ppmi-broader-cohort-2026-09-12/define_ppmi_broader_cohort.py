import os
import json
from pathlib import Path
from collections import Counter

private_config=json.loads(Path(os.environ['PPMI_PRIVATE_CONFIG']).read_text())
source=Path('outputs/ppmi-ir3-eligibility-audit/sample_evidence.json')
rows=json.loads(source.read_text())
out=Path('outputs/ppmi-broader-cohort-2026-09-12');out.mkdir(exist_ok=True)
selected=[]
for r in rows:
    if r['visit']!='BL' or r['release_QC']!='pass':continue
    if r['metadata_label']=='PD' and r['diagnosis']=='PD_supported':group='PD'
    elif r['metadata_label']=='Control' and r['diagnosis']=='no_neurological_disorder_supported':group='Control'
    else:continue
    assert len(r['count_files'])==1 and r['sample_link']=='linked_at_participant_visit_month'
    item=dict(r)
    item.update(analysis_group=group,cohort_status='PROVISIONALLY_INCLUDED_PENDING_EXPRESSION_QC',original_strict_audit_decision=r['decision'])
    item.pop('decision')
    item['exclude_in_medication_timing_sensitivity']=group=='PD' and r['PATNO'] in set(map(str, private_config['medication_timing_exclusions']))
    selected.append(item)
assert Counter(x['analysis_group'] for x in selected)=={'PD':393,'Control':186}
assert len({x['PATNO'] for x in selected})==579
assert sum(x['exclude_in_medication_timing_sensitivity'] for x in selected)==4
(out/'cohort_manifest.json').write_text(json.dumps(selected,indent=2))
summary={'approved_date':'2026-09-12','status':'provisional_pending_independent_expression_QC','primary':{'PD':393,'Control':186,'total':579},'medication_timing_sensitivity':{'PD':389,'Control':186,'total':575},'excluded_control_PATNO':private_config['excluded_control_ids'],'timing_sensitivity_exclusions':private_config['medication_timing_exclusions'],'definition':'PD participants reporting no PD medication at collection versus PPMI control participants; lifetime never-treated status and otherwise-healthy control status are not claimed.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
