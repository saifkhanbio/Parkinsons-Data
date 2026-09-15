"""Apply one saved fold calibration map to that model's original probabilities."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import pandas as pd
from calibrator import apply
p=argparse.ArgumentParser()
p.add_argument('--calibrator',required=True,help='JSON for the corresponding RNA model/fold')
p.add_argument('--predictions',required=True,help='TSV: PATNO and probability_PD')
p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output)
if out.exists():raise SystemExit('Output already exists')
state=json.loads(Path(a.calibrator).read_text());d=pd.read_csv(a.predictions,sep='\t',dtype={'PATNO':str})
assert d.PATNO.is_unique
for k in ['protocol','repeat','fold']:
 if k in d:assert d[k].eq(state[k]).all(),f'Mismatched {k}'
d['original_probability_PD']=d.probability_PD
d['probability_PD']=apply(state,d.original_probability_PD.to_numpy())
if 'threshold' in d:d['threshold']=apply(state,d.threshold.to_numpy())
d.to_csv(out,sep='\t',index=False)
