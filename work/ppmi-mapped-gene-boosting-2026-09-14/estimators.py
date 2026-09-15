"""Fixed alternative estimators on the unchanged mapped-gene representation."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "dependencies"))
import warnings
import tempfile
import joblib
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
    'xgboost': [dict(max_depth=d, reg_lambda=l2) for d in [2, 3, 4] for l2 in [10., 1.]],
    'catboost': [dict(depth=d, l2_leaf_reg=l2) for d in [2, 3, 4] for l2 in [10., 1.]],
}
ALGORITHMS = list(GRIDS)


def fit(x, y, algorithm, parameters):
    x = np.asarray(x, dtype=np.float32)
    if algorithm == 'xgboost':
        model = XGBClassifier(**parameters, n_estimators=200, learning_rate=.05,
            min_child_weight=5, subsample=.8, colsample_bytree=.8, max_bin=64,
            tree_method='hist', device='cuda:0', objective='binary:logistic',
            eval_metric='logloss', n_jobs=2, random_state=SEED)
    elif algorithm == 'catboost':
        model = CatBoostClassifier(**parameters, iterations=200, learning_rate=.05,
            loss_function='Logloss', task_type='GPU', devices='0', thread_count=2,
            border_count=64, boosting_type='Plain', bootstrap_type='Bernoulli',
            subsample=.8, random_seed=SEED, random_strength=1.,
            gpu_ram_part=.5, allow_writing_files=False, verbose=False)
    else:
        raise ValueError(algorithm)
    model.fit(x, y)
    if algorithm == 'xgboost':
        import json
        config = json.loads(model.get_booster().save_config())
        assert config['learner']['generic_param']['device'].startswith('cuda'), config
    else:
        assert model.get_all_params()['task_type'] == 'GPU'
    model.convergence_retries_ = 0
    assert np.array_equal(model.classes_, [0, 1])
    return model


def score(model, x, algorithm):
    x = np.asarray(x, dtype=np.float32)
    if algorithm == 'xgboost':
        from xgboost import DMatrix
        values = model.get_booster().predict(DMatrix(x, nthread=2))
    else:
        values = model.predict_proba(x)[:, 1]
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
    import time
    checks = validate_mapping(device)
    assert device.type == 'cuda', 'GPU required for this experiment'
    rng = np.random.default_rng(SEED)
    raw = rng.poisson(30, (100, 100)).astype(np.int32)
    y = np.tile([0, 1], 50)
    raw[y == 1, :5] += 25
    genes, indices = np.arange(100).astype(str), np.arange(80)
    tr, te = np.arange(80), np.arange(80, 100)
    data = MappedData(raw, genes, y, indices, device)
    pack = data.pack(tr, te)
    # Check all training inputs remain identical after held-out alteration.
    altered = raw.copy(); altered[te] += 10000
    labels = y.copy(); labels[te] = 1 - labels[te]
    changed = MappedData(altered, genes, labels, indices, device)
    splits = make_splits(y[tr], np.arange(80), 'participant_stratified', 3, SEED)
    for a, b in splits:
        original_pack = data.pack(tr[a], tr[b])
        changed_pack = changed.pack(tr[a], tr[b])
        for key in ['train', 'test']:
            np.testing.assert_array_equal(original_pack[key], changed_pack[key])
        np.testing.assert_array_equal(data.labels[tr[a]], changed.labels[tr[a]])
        np.testing.assert_array_equal(data.labels[tr[b]], changed.labels[tr[b]])
    with tempfile.TemporaryDirectory(prefix='boosting_validation_') as folder:
        for algorithm in ALGORITHMS:
            start = time.monotonic()
            params = GRIDS[algorithm][0]
            model = fit(pack['train'], y[tr], algorithm, params)
            state = artifact(model, pack, dict(algorithm=algorithm, parameters=params), genes)
            values = score(model, pack['test'], algorithm)
            assert roc_auc_score(y[te], values) > .9
            path = Path(folder) / (algorithm + '.joblib')
            joblib.dump(state, path)
            np.testing.assert_allclose(predict_artifact(joblib.load(path), raw[te]), values, atol=1e-7)
            checks[algorithm] = dict(GPU_fit='PASS', inference='PASS', seconds=time.monotonic()-start)
    checks['boosting_training_input_holdout_invariance'] = 'PASS'
    return checks
