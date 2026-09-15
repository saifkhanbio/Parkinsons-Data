"""Apply a trusted SVM development artifact; scores are uncalibrated margins."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from svm_analysis import predict_artifact

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',required=True)
parser.add_argument('--counts',required=True,help='Full matching Geneid-by-participant raw-count TSV, optionally gzip')
parser.add_argument('--metadata',required=True,help='PATNO and required covariates in TSV')
parser.add_argument('--output',required=True)
args=parser.parse_args()
out=Path(args.output)
assert not out.exists(),'Refusing to overwrite predictions'
state=joblib.load(args.model)
assert state['algorithm']=='rbf_svm'
meta=pd.read_csv(args.metadata,sep='\t',dtype={'PATNO':str})
counts=pd.read_csv(args.counts,sep='\t').set_index('Geneid')
assert meta.PATNO.is_unique and counts.index.is_unique and counts.columns.is_unique
assert set(counts.index)==set(state['feature_universe']) and set(meta.PATNO)<=set(counts.columns)
raw=counts.loc[state['feature_universe'],meta.PATNO].to_numpy().T
margin=predict_artifact(state,raw,meta)
pd.DataFrame(dict(PATNO=meta.PATNO,uncalibrated_PD_decision_margin=margin,
                  predicted_PD_native_boundary=margin>=0)).to_csv(out,sep='\t',index=False)
print(f'Saved {len(meta)} uncalibrated SVM scores to {out}')
