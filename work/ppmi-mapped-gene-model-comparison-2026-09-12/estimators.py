"""Fixed alternative estimators on the unchanged mapped-gene representation."""
from pathlib import Path
import sys
import warnings
import tempfile
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

WORK = Path(__file__).resolve().parent.parent
MAPPED = WORK / 'ppmi-mapped-gene-classifier-2026-09-12'
PREVIOUS = WORK / 'ppmi-mapped-gene-svm-2026-09-12'
sys.path.insert(0, str(MAPPED))
from mapped_genes import BASE, PRIOR, HALLMARK, MappedData, choose_device, make_splits, SEED, validate as validate_mapping

GRIDS = {
    'linear_svm': [dict(C=c) for c in [.0001, .001, .01, .1, 1.]],
    'extra_trees': [dict(min_samples_leaf=leaf, max_features=features)
                    for leaf in [30, 15, 5] for features in ['sqrt', .1]],
    'hist_gradient_boosting': [dict(max_leaf_nodes=leaf, l2_regularization=l2)
                               for leaf in [3, 7] for l2 in [10., 1.]],
}
ALGORITHMS = list(GRIDS)


def fit(x, y, algorithm, parameters):
    if algorithm == 'linear_svm':
        model = LinearSVC(**parameters, penalty='l2', loss='squared_hinge', dual=True,
                          tol=1e-5, max_iter=20000, random_state=SEED)
    elif algorithm == 'extra_trees':
        model = ExtraTreesClassifier(**parameters, n_estimators=300, bootstrap=False,
                                     max_depth=None, n_jobs=2, random_state=SEED)
    elif algorithm == 'hist_gradient_boosting':
        model = HistGradientBoostingClassifier(**parameters, learning_rate=.05, max_iter=150,
                    min_samples_leaf=20, max_bins=63, max_features=1., early_stopping=False, random_state=SEED)
    else:
        raise ValueError(algorithm)
    retries = 0
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            model.fit(x, y)
        except ConvergenceWarning:
            if algorithm != 'linear_svm':
                raise
            retries = 1
            model.set_params(max_iter=100000).fit(x, y)
    model.convergence_retries_ = retries
    assert np.array_equal(model.classes_, [0, 1])
    if algorithm == 'linear_svm':
        assert model.n_iter_ < model.max_iter and np.isfinite(model.coef_).all()
    return model


def score(model, x, algorithm):
    values = model.decision_function(x) if algorithm == 'linear_svm' else model.predict_proba(x)[:, 1]
    assert np.isfinite(values).all()
    return values


def tune(data, train, splits, grids=None, progress=None):
    grids = GRIDS if grids is None else grids
    values = {(a, i): [] for a, candidates in grids.items() for i in range(len(candidates))}
    retries = {key: 0 for key in values}
    seen = np.zeros(len(train), dtype=int)
    for inner_number, (a, b) in enumerate(splits, 1):
        assert not np.intersect1d(a, b).size
        seen[b] += 1
        pack = data.pack(train[a], train[b])
        for algorithm, candidates in grids.items():
            if progress:
                progress(inner_number, algorithm)
            for i, parameters in enumerate(candidates):
                model = fit(pack['train'], data.labels[train[a]], algorithm, parameters)
                values[(algorithm, i)].append(float(roc_auc_score(data.labels[train[b]], score(model, pack['test'], algorithm))))
                retries[(algorithm, i)] += model.convergence_retries_
    assert (seen == 1).all()
    rows = [dict(algorithm=algorithm, candidate=i, parameters=grids[algorithm][i],
                 mean_inner_AUROC=float(np.mean(v)), inner_AUROCs=';'.join(map(str, v)),
                 convergence_retries=retries[(algorithm, i)]) for (algorithm, i), v in values.items()]
    best = {a: min([r for r in rows if r['algorithm'] == a], key=lambda r: (-r['mean_inner_AUROC'], r['candidate'])).copy() for a in grids}
    return best, rows


def artifact(model, pack, chosen, genes):
    algorithm = chosen['algorithm']
    return dict(classifier=model, algorithm=algorithm, hyperparameters=chosen, rna=pack['state'],
                feature_universe=np.asarray(genes), selected_gene_ids=np.asarray(genes)[pack['state']['indices']],
                positive_class='PD', score_type='uncalibrated_margin' if algorithm == 'linear_svm' else 'uncalibrated_probability_estimate',
                threshold=0. if algorithm == 'linear_svm' else .5)


def predict_artifact(state, raw):
    raw = np.asarray(raw)
    assert raw.ndim == 2 and raw.shape[1] == len(state['feature_universe'])
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw == np.floor(raw)).all()
    library = raw.sum(1, dtype=np.float64)
    assert (library > 0).all()
    s = state['rna']
    values = np.log2(1 + raw[:, s['indices']] / library[:, None] * 1e6)
    return score(state['classifier'], (values - s['means']) / s['scales'], state['algorithm'])


def validate(device):
    checks = validate_mapping(device)
    rng = np.random.default_rng(SEED)
    raw = rng.poisson(30, (100, 100)).astype(np.int32)
    y = np.tile([0, 1], 50)
    raw[y == 1, :5] += 20
    genes, indices = np.arange(100).astype(str), np.arange(80)
    tr, te = np.arange(80), np.arange(80, 100)
    data = MappedData(raw, genes, y, indices, device)
    splits = make_splits(y[tr], np.arange(80), 'participant_stratified', 3, SEED)
    small = {a: [v[0], v[-1]] for a, v in GRIDS.items()}
    best, rows = tune(data, tr, splits, small)
    altered, labels = raw.copy(), y.copy()
    altered[te] += 10000
    labels[te] = 1 - labels[te]
    changed = MappedData(altered, genes, labels, indices, device)
    other, other_rows = tune(changed, tr, splits, small)
    assert best == other and rows == other_rows
    pack = data.pack(tr, te)
    with tempfile.TemporaryDirectory(prefix='mapped_models_validation_') as folder:
        for a, chosen in best.items():
            model = fit(pack['train'], y[tr], a, chosen['parameters'])
            state = artifact(model, pack, chosen, genes)
            path = Path(folder) / (a + '.joblib')
            joblib.dump(state, path)
            np.testing.assert_allclose(predict_artifact(joblib.load(path), raw[te]), score(model, pack['test'], a), atol=1e-9)
    centers = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    x = np.repeat(centers, 80, axis=0) + rng.normal(0, .15, (320, 2))
    labels = np.repeat([0, 1, 1, 0], 80)
    tr = np.concatenate([np.arange(j * 80, j * 80 + 60) for j in range(4)])
    te = np.setdiff1d(np.arange(320), tr)
    for a in ['extra_trees', 'hist_gradient_boosting']:
        parameters = GRIDS[a][-1].copy()
        if a == 'extra_trees':
            parameters['max_features'] = 'sqrt'
        model = fit(x[tr], labels[tr], a, parameters)
        auc = float(roc_auc_score(labels[te], score(model, x[te], a)))
        assert auc > .9, (a, auc)
        checks[a + '_synthetic_nonlinear_AUROC'] = auc
    checks.update(all_model_holdout_tuning='PASS', all_model_serialized_inference='PASS')
    return checks
