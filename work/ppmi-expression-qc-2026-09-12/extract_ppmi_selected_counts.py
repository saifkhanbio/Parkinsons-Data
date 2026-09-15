import os
import tarfile,json,time,subprocess,io
from pathlib import Path
import pandas as pd
import numpy as np

out=Path('outputs/ppmi-expression-qc-2026-09-12');out.mkdir(exist_ok=True)
cohort=json.loads(Path('outputs/ppmi-broader-cohort-2026-09-12/cohort_manifest.json').read_text())
targets={r['count_files'][0]:r for r in cohort}
status={'stage':'extracting_selected_counts','selected_total':len(targets),'extracted':0}
def save(): (out/'status.json').write_text(json.dumps(status,indent=2))
save(); start=time.time();last=start;columns={}; genes=None
try:
    with tarfile.open(os.environ['PPMI_IR3_ARCHIVE'],'r|gz') as t:
        for m in t:
            if m.name in targets:
                d=pd.read_csv(io.BytesIO(t.extractfile(m).read()),sep='\t',comment='#',usecols=[0,5,6])
                assert d.iloc[:,0].is_unique
                if genes is None:
                    genes=d.iloc[:,0].tolist()
                    d.iloc[:,:2].to_csv(out/'gene_annotation.tsv',sep='\t',index=False)
                assert genes==d.iloc[:,0].tolist(),m.name
                values=d.iloc[:,2].to_numpy()
                assert np.isfinite(values).all() and (values>=0).all() and (values==np.floor(values)).all()
                columns[targets[m.name]['PATNO']]=values.astype(np.int64)
                status['extracted']=len(columns)
            if time.time()-last>=25:
                last=time.time();status.update(elapsed_seconds=round(last-start),last_archive_member=m.name);save();print(json.dumps(status),flush=True)
            if len(columns)==len(targets):break
    assert len(columns)==579
    order=[r['PATNO'] for r in cohort]
    pd.DataFrame({k:columns[k] for k in order},index=pd.Index(genes,name='Geneid')).to_csv(out/'raw_counts.tsv.gz',sep='\t',compression='gzip')
    metadata=pd.read_csv('outputs/ppmi-ir3-archive-check/metaDataIR3.csv',dtype=str).set_index('Sample')
    records=[]
    for r in cohort:
        v=metadata.loc[r['sample']].to_dict();v.update(Sample=r['sample'],PATNO=r['PATNO'],group=r['analysis_group'],timing_sensitivity_exclude=r['exclude_in_medication_timing_sensitivity']);records.append(v)
    pd.DataFrame(records).to_csv(out/'sample_metadata.tsv',sep='\t',index=False)
    status.update(stage='counts_ready',genes=len(genes),elapsed_seconds=round(time.time()-start));save()
    print(json.dumps(status),flush=True)
except Exception as e:
    status.update(stage='FAILED',error=repr(e));save();raise
