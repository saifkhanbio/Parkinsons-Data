"""Nested elastic-net/ridge refinement with training-only operating thresholds."""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

BASE = Path(__file__).resolve().parent.parent / 'ppmi-classifier-2026-09-12'
sys.path.insert(0, str(BASE))
from classifier import (FoldData, choose_device, make_splits, matrices, artifact,
                        predict_artifact, fit_logistic, metrics, validate_leakage,
                        CS, KS, SEED, CLINICAL_FEATURES)

FAMILIES = ['blood_clinical', 'rna', 'combined']
PENALTIES = ['ridge', 'elasticnet']
RATIOS = [.1, .5, .9]


def fit_model(x, y, c, penalty, ratio):
    if penalty == 'ridge':
        return fit_logistic(x, y, c)
    assert penalty == 'elasticnet'
    model = LogisticRegression(C=c, penalty='elasticnet', solver='saga', l1_ratio=ratio,
                               max_iter=3000, tol=1e-5, random_state=SEED, warm_start=True)
    iterations = 0
    for limit in [3000, 12000]:
        model.max_iter = limit
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y)
        iterations += int(model.n_iter_[0])
        if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
            assert model.classes_.tolist() == [0, 1] and np.isfinite(model.coef_).all()
            model.total_optimizer_iterations_ = iterations
            return model
    raise RuntimeError(f'SAGA did not converge: C={c}, l1_ratio={ratio}, features={x.shape[1]}')


def select_threshold(y, p):
    y, p = np.asarray(y), np.asarray(p)
    assert set(np.unique(y)) == {0, 1} and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    # sklearn uses infinity for all-negative; a finite value above max(p) is equivalent.
    thresholds[~np.isfinite(thresholds)] = np.nextafter(p.max(), np.inf)
    ba, gap = (tpr + 1 - fpr) / 2, np.abs(tpr - (1 - fpr))
    candidates = np.flatnonzero(np.isclose(ba, ba.max(), atol=1e-12, rtol=0))
    chosen = min(candidates, key=lambda i: (gap[i], abs(thresholds[i] - .5), thresholds[i]))
    return float(thresholds[chosen]), float(ba[chosen])


def tune_models(data, train, splits, families=None, cs=None, ks=None, ratios=None):
    families = FAMILIES if families is None else families
    cs, ks, ratios = (CS if cs is None else cs), (KS if ks is None else ks), (RATIOS if ratios is None else ratios)
    keys = [(f, penalty, k, c, ratio) for f in families for penalty in PENALTIES
            for k in (ks if f in ['rna', 'combined'] else [0]) for c in cs
            for ratio in ([0.] if penalty == 'ridge' else ratios)]
    probabilities = {key: np.full(len(train), np.nan) for key in keys}
    scores = {key: [] for key in keys}
    iterations = {key: [] for key in keys}
    seen = np.zeros(len(train), dtype=int)
    for a, b in splits:
        assert not np.intersect1d(a, b).size
        seen[b] += 1
        tr, va = train[a], train[b]
        pack = data.pack(tr, va)
        for key in keys:
            f, penalty, k, c, ratio = key
            x, v = matrices(pack, f, k)
            model = fit_model(x, data.labels[tr], c, penalty, ratio)
            p = model.predict_proba(v)[:, 1]
            probabilities[key][b] = p
            scores[key].append(float(roc_auc_score(data.labels[va], p)))
            iterations[key].append(int(getattr(model, 'total_optimizer_iterations_', model.n_iter_[0])))
    assert (seen == 1).all()
    rows = []
    for key in keys:
        f, penalty, k, c, ratio = key
        assert np.isfinite(probabilities[key]).all()
        rows.append(dict(family=f, penalty=penalty, k=k, C=c, l1_ratio=ratio,
                         mean_inner_AUROC=float(np.mean(scores[key])),
                         inner_AUROCs=';'.join(map(str, scores[key])),
                         maximum_optimizer_iterations=max(iterations[key])))
    chosen = {}
    for f in families:
        for penalty in PENALTIES:
            candidates = [r for r in rows if r['family'] == f and r['penalty'] == penalty]
            best = min(candidates, key=lambda r: (-r['mean_inner_AUROC'], r['k'], r['C'], -r['l1_ratio'])).copy()
            key = (f, penalty, best['k'], best['C'], best['l1_ratio'])
            threshold, ba = select_threshold(data.labels[train], probabilities[key])
            best.update(threshold=threshold, inner_threshold_balanced_accuracy=ba)
            chosen[(f, penalty)] = best
    return chosen, rows


def evaluate(y, p, threshold):
    result = metrics(y, p)
    predicted = p >= threshold
    result.update(balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                  sensitivity=float(predicted[y == 1].mean()),
                  specificity=float((~predicted[y == 0]).mean()),
                  accuracy=float((predicted == y).mean()),
                  TP=int(predicted[y == 1].sum()), FN=int((~predicted[y == 1]).sum()),
                  TN=int((~predicted[y == 0]).sum()), FP=int(predicted[y == 0].sum()))
    return result


def validate_refinement(device):
    checks = validate_leakage(device)
    rng = np.random.default_rng(SEED)
    # Brute-force oracle verifies threshold behavior with tied and constant predictions.
    for p in [np.array([.2, .3, .3, .7, .9, .9]), np.repeat(.68, 6)]:
        y = np.array([0, 0, 1, 0, 1, 1])
        threshold, ba = select_threshold(y, p)
        expected = max(balanced_accuracy_score(y, p >= t) for t in np.r_[np.unique(p), np.nextafter(p.max(), np.inf)])
        np.testing.assert_allclose(ba, expected)
        np.testing.assert_allclose(ba, balanced_accuracy_score(y, p >= threshold))
    # Perturb only outer-held-out data: hyperparameters and thresholds must be unchanged.
    n = 72
    raw = rng.poisson(30, (n, 120)).astype(np.int32)
    y = np.tile([0, 1], n // 2)
    raw[y == 1, :5] += 12
    meta = pd.DataFrame(dict(age_collection_years=rng.normal(60, 8, n), sex=np.where(y, 'Male', 'Female'),
                            wbc=rng.uniform(4, 8, n), neutrophils_percent=rng.uniform(40, 70, n),
                            monocytes_percent=rng.uniform(3, 8, n), eosinophils_percent=rng.uniform(1, 3, n),
                            basophils_percent=rng.uniform(.1, 1, n), RIN=rng.uniform(6, 9, n),
                            intergenic_percent=rng.uniform(5, 15, n), phase='PPMI-Phase1'))
    genes = np.arange(120).astype(str)
    tr, te = np.arange(54), np.arange(54, n)
    splits = make_splits(y[tr], np.arange(len(tr)), 'participant_stratified', 3, SEED)
    data = FoldData(raw, genes, meta, y, device)
    best, _ = tune_models(data, tr, splits, families=['combined'], cs=[.01, .1], ks=[100], ratios=[.5])
    altered_raw, altered_y, altered_meta = raw.copy(), y.copy(), meta.copy()
    altered_raw[te] += 10000
    altered_y[te] = 1 - altered_y[te]
    altered_meta.loc[te, 'age_collection_years'] += 1000
    other = FoldData(altered_raw, genes, altered_meta, altered_y, device)
    altered_best, _ = tune_models(other, tr, splits, families=['combined'], cs=[.01, .1], ks=[100], ratios=[.5])
    assert best == altered_best
    pack = data.pack(tr, te)
    b = best[('combined', 'elasticnet')]
    x, v = matrices(pack, 'combined', b['k'])
    model = fit_model(x, y[tr], b['C'], 'elasticnet', b['l1_ratio'])
    state = artifact(model, pack, 'combined', b['k'], genes)
    state['threshold'] = b['threshold']
    p = predict_artifact(state, raw[te], meta.iloc[te])
    np.testing.assert_allclose(p, model.predict_proba(v)[:, 1], atol=1e-9)
    assert np.count_nonzero(model.coef_) < model.coef_.size
    checks.update(threshold_brute_force='PASS', outer_perturbation_tuning_and_threshold='PASS',
                  elasticnet_inference_and_sparsity='PASS')
    return checks
