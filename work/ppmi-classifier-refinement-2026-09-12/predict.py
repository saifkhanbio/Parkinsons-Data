"""Apply a trusted refinement artifact with its saved training-selected threshold."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from refine import predict_artifact

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--counts', required=True, help='Full matching Geneid-by-participant raw-count TSV, optionally gzip')
parser.add_argument('--metadata', required=True, help='PATNO and required covariates in TSV')
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
assert not out.exists(), 'Refusing to overwrite existing predictions'
state = joblib.load(args.model)
meta = pd.read_csv(args.metadata, sep='\t', dtype={'PATNO': str})
counts = pd.read_csv(args.counts, sep='\t').set_index('Geneid')
assert meta.PATNO.is_unique and counts.index.is_unique and counts.columns.is_unique
assert set(counts.index) == set(state['feature_universe']), 'Require full matching gene universe for CPM'
assert set(meta.PATNO) <= set(counts.columns)
raw = counts.loc[state['feature_universe'], meta.PATNO].to_numpy().T
p = predict_artifact(state, raw, meta)
pd.DataFrame(dict(PATNO=meta.PATNO, probability_PD=p, threshold=state['threshold'],
                  predicted_PD=p >= state['threshold'])).to_csv(out, sep='\t', index=False)
print(f'Saved {len(meta)} predictions to {out}')
