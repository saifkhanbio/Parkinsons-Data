"""Apply a trusted local development artifact to a matching raw-count matrix."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from classifier import predict_artifact

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True, help='Trusted local models/*.joblib artifact')
parser.add_argument('--counts', required=True, help='Geneid-by-participant TSV, optionally gzip compressed')
parser.add_argument('--metadata', required=True, help='TSV with PATNO and covariates needed by the chosen family')
parser.add_argument('--output', required=True)
args = parser.parse_args()
state = joblib.load(args.model)
meta = pd.read_csv(args.metadata, sep='\t', dtype={'PATNO': str})
counts = pd.read_csv(args.counts, sep='\t').set_index('Geneid')
assert meta.PATNO.is_unique and counts.index.is_unique and counts.columns.is_unique
assert set(counts.index) == set(state['feature_universe']), 'Require the complete matching annotation universe for CPM normalization'
assert set(meta.PATNO) <= set(counts.columns)
raw = counts.loc[state['feature_universe'], meta.PATNO].to_numpy().T
probabilities = predict_artifact(state, raw, meta)
out = Path(args.output)
assert not out.exists(), 'Refusing to overwrite an existing prediction file'
pd.DataFrame({'PATNO': meta.PATNO, 'probability_PD': probabilities,
              'predicted_PD_threshold_0_5': probabilities >= .5}).to_csv(out, sep='\t', index=False)
print(f'Saved {len(meta)} development-model predictions to {out}')
