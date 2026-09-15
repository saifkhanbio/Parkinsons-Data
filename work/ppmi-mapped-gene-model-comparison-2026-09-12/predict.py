"""Apply a trusted local model to a TSV: sample ID, then complete raw Geneid columns."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from estimators import predict_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'counts', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit('Refusing to overwrite existing output')
    state = joblib.load(args.model)
    frame = pd.read_csv(args.counts, sep='\t', index_col=0)
    expected = state['feature_universe'].tolist()
    assert frame.index.is_unique and frame.columns.is_unique
    assert len(frame.columns) == len(expected) and set(frame.columns) == set(expected)
    values = predict_artifact(state, frame.loc[:, expected].to_numpy())
    pd.DataFrame(dict(sample_id=frame.index, score=values, score_type=state['score_type'],
        predicted_PD_native=(values >= state['threshold']).astype(int))).to_csv(args.output, sep='\t', index=False)


if __name__ == '__main__':
    main()
