"""Apply a trusted local model with full count universe and aligned technical QC."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,joblib
import pandas as pd
from adjustment import predict
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--counts',required=True);p.add_argument('--metadata',help='TSV PATNO and intergenic_percent; required for adjusted model');p.add_argument('--output',required=True);a=p.parse_args()
if Path(a.output).exists():raise SystemExit('Output already exists')
s=joblib.load(a.model);counts=pd.read_csv(a.counts,sep='\t').set_index('Geneid');assert counts.index.is_unique and counts.columns.is_unique and set(counts.index)==set(s['feature_universe'])
ids=counts.columns.to_numpy();qc=None
if s['adjustment'] is not None:
 if not a.metadata:raise SystemExit('Adjusted model requires technical metadata')
 meta=pd.read_csv(a.metadata,sep='\t',dtype={'PATNO':str}).set_index('PATNO');assert meta.index.is_unique and set(meta.index)==set(ids);qc=meta.loc[ids,'intergenic_percent'].to_numpy(float)
raw=counts.loc[s['feature_universe'],ids].to_numpy().T;p=predict(s,raw,qc)
pd.DataFrame(dict(PATNO=ids,probability_PD=p,threshold=s['threshold'],predicted_PD=p>=s['threshold'])).to_csv(a.output,sep='\t',index=False)
