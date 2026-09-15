"""Fine C search with an exactly reproducible coarse-grid control."""
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

MAPPED = Path(__file__).resolve().parent.parent / 'ppmi-mapped-gene-classifier-2026-09-12'
sys.path.insert(0, str(MAPPED))
from mapped_genes import (BASE, PRIOR, HALLMARK, MappedData, choose_device, make_splits, SEED,
    fit_model, select_threshold, evaluate, artifact, predict_artifact, validate as validate_mapping)

PENALTIES = ['ridge']
COARSE_CS = [.001, .01, .1, 1.]
FINE_CS = [.0001, .0003, .001, .003, .01, .03, .1, .3, 1., 3., 10.]


def tune(data, train, splits, cs=None):
    cs = FINE_CS if cs is None else cs
    predictions = {c: np.full(len(train), np.nan) for c in cs}
    scores = {c: [] for c in cs}
    seen = np.zeros(len(train), dtype=int)
    counts = []
    for a, b in splits:
        assert not np.intersect1d(a, b).size
        seen[b] += 1
        pack = data.pack(train[a], train[b])
        counts.append(pack['train'].shape[1])
        for c in cs:
            model = fit_model(pack['train'], data.labels[train[a]], c, 'ridge', 0.)
            p = model.predict_proba(pack['test'])[:, 1]
            assert np.isfinite(p).all()
            predictions[c][b] = p
            scores[c].append(float(roc_auc_score(data.labels[train[b]], p)))
    assert (seen == 1).all()
    rows = [dict(penalty='ridge', C=c, l1_ratio=0., mean_inner_AUROC=float(np.mean(v)),
                 inner_AUROCs=';'.join(map(str, v)), inner_eligible_counts=';'.join(map(str, counts))) for c, v in scores.items()]
    chosen = min(rows, key=lambda r: (-r['mean_inner_AUROC'], r['C'])).copy()
    threshold, ba = select_threshold(data.labels[train], predictions[chosen['C']])
    chosen.update(threshold=threshold, inner_threshold_balanced_accuracy=ba)
    coarse_rows = [r for r in rows if r['C'] in COARSE_CS]
    if len(coarse_rows) == len(COARSE_CS):
        coarse = min(coarse_rows, key=lambda r: (-r['mean_inner_AUROC'], r['C']))
        coarse_threshold, _ = select_threshold(data.labels[train], predictions[coarse['C']])
        chosen.update(coarse_C=coarse['C'], coarse_threshold=coarse_threshold,
                      coarse_mean_inner_AUROC=coarse['mean_inner_AUROC'])
    return {'ridge': chosen}, rows


def validate(device):
    checks = validate_mapping(device)
    rng = np.random.default_rng(SEED)
    raw = rng.poisson(30, (80, 250)).astype(np.int32)
    y = np.tile([0, 1], 40)
    raw[y == 1, :5] += 12
    genes, indices = np.arange(250).astype(str), np.arange(180)
    tr, te = np.arange(60), np.arange(60, 80)
    data = MappedData(raw, genes, y, indices, device)
    splits = make_splits(y[tr], np.arange(60), 'participant_stratified', 3, SEED)
    best, rows = tune(data, tr, splits)
    altered, labels = raw.copy(), y.copy()
    altered[te] += 10000
    labels[te] = 1 - labels[te]
    changed = MappedData(altered, genes, labels, indices, device)
    other, other_rows = tune(changed, tr, splits)
    assert best == other and rows == other_rows
    coarse, _ = tune(data, tr, splits, cs=COARSE_CS)
    assert best['ridge']['coarse_C'] == coarse['ridge']['C']
    assert best['ridge']['coarse_threshold'] == coarse['ridge']['threshold']
    assert best['ridge']['mean_inner_AUROC'] >= coarse['ridge']['mean_inner_AUROC']
    pack = data.pack(tr, te)
    chosen = best['ridge']
    model = fit_model(pack['train'], y[tr], chosen['C'], 'ridge', 0.)
    state = artifact(model, pack, chosen, genes)
    np.testing.assert_allclose(predict_artifact(state, raw[te]), model.predict_proba(pack['test'])[:, 1], atol=1e-9)
    checks.update(fine_grid_holdout_tuning_threshold='PASS', coarse_grid_control='PASS', fine_grid_inference='PASS')
    return checks
