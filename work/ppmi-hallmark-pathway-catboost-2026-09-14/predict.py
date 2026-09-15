"""Apply the local pathway model to complete matching raw counts."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
from run import predict
import joblib
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'counts', 'output']: parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if Path(args.output).exists(): raise SystemExit('Refusing to overwrite output')
    state = joblib.load(args.model)
    counts = pd.read_csv(args.counts, sep='\t', index_col=0)
    genes = state['feature_universe'].tolist()
    assert counts.index.is_unique and counts.columns.is_unique
    assert set(counts.columns) == set(genes) and len(counts.columns) == len(genes)
    probabilities = predict(state, counts.loc[:, genes].to_numpy())
    pd.DataFrame(dict(sample_id=counts.index, probability=probabilities,
                     predicted_PD=(probabilities >= .5).astype(int))).to_csv(args.output, sep='\t', index=False)
