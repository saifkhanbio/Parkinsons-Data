"""Apply a trusted Hallmark classifier artifact and its saved training transform."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from pathways import predict_artifact

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True); p.add_argument('--counts',required=True)
p.add_argument('--metadata',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); out=Path(a.output)
assert not out.exists(),'Refusing to overwrite predictions'
state=joblib.load(a.model)
meta=pd.read_csv(a.metadata,sep='\t',dtype={'PATNO':str})
counts=pd.read_csv(a.counts,sep='\t').set_index('Geneid')
assert meta.PATNO.is_unique and counts.index.is_unique and counts.columns.is_unique
assert set(counts.index)==set(state['feature_universe']) and set(meta.PATNO)<=set(counts.columns)
raw=counts.loc[state['feature_universe'],meta.PATNO].to_numpy().T
prob=predict_artifact(state,raw,meta)
pd.DataFrame(dict(PATNO=meta.PATNO,probability_PD=prob,threshold=state['threshold'],predicted_PD=prob>=state['threshold'])).to_csv(out,sep='\t',index=False)
print(f'Saved {len(meta)} development predictions to {out}')
