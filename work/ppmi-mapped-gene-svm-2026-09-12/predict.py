"""Apply a trusted local SVM artifact to complete, matching raw gene counts.

Counts TSV: first column sample ID, remaining columns exact versioned Geneids.
Column order may differ; the complete saved raw gene universe is required.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from mapped_svm import predict_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--counts', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit('Refusing to overwrite existing output')
    state = joblib.load(args.model)
    frame = pd.read_csv(args.counts, sep='\t', index_col=0)
    expected = state['feature_universe'].tolist()
    assert frame.index.is_unique and frame.columns.is_unique
    assert len(frame.columns) == len(expected) and set(frame.columns) == set(expected)
    margin = predict_artifact(state, frame.loc[:, expected].to_numpy())
    pd.DataFrame(dict(sample_id=frame.index, decision_margin=margin,
                      predicted_PD_native=(margin >= 0).astype(int))).to_csv(args.output, sep='\t', index=False)


if __name__ == '__main__':
    main()
