"""Apply a trusted mapped-RNA artifact using its complete raw-count denominator."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from ridge_tuning import predict_artifact

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True); p.add_argument('--counts',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); out=Path(a.output)
assert not out.exists(),'Refusing to overwrite predictions'
state=joblib.load(a.model); counts=pd.read_csv(a.counts,sep='\t').set_index('Geneid')
assert counts.index.is_unique and counts.columns.is_unique and set(counts.index)==set(state['feature_universe'])
raw=counts.loc[state['feature_universe']].to_numpy().T
prob=predict_artifact(state,raw)
pd.DataFrame(dict(PATNO=counts.columns,probability_PD=prob,threshold=state['threshold'],predicted_PD=prob>=state['threshold'])).to_csv(out,sep='\t',index=False)
print(f'Saved {len(prob)} RNA-only predictions to {out}')
