"""Predict using a trusted saved score model and aligned full-panel raw counts."""
import argparse,sys
sys.dont_write_bytecode=True
from pathlib import Path
import joblib,pandas as pd
from path_models import predict_artifact

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--counts',required=True,help='TSV: Geneid rows and participant columns, all original features');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);assert not out.exists(),'Refusing overwrite'
    state=joblib.load(a.model);counts=pd.read_csv(a.counts,sep='\t').set_index('Geneid')
    assert counts.index.is_unique and counts.columns.is_unique and set(counts.index)==set(state['feature_universe'])
    probability=predict_artifact(state,counts.loc[state['feature_universe']].to_numpy().T)
    pd.DataFrame(dict(PATNO=counts.columns,probability_PD=probability,threshold=state['threshold'],predicted_PD=probability>=state['threshold'])).to_csv(out,sep='\t',index=False)
