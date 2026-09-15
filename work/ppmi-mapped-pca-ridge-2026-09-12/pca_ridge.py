"""Training-only PCA with a preserved gene-space ridge control."""
from pathlib import Path
import sys
import tempfile
import joblib
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

MAPPED = Path(__file__).resolve().parent.parent/'ppmi-mapped-gene-classifier-2026-09-12'
sys.path.insert(0, str(MAPPED))
from mapped_genes import (BASE, PRIOR, HALLMARK, MappedData, choose_device, make_splits,
    SEED, fit_model, select_threshold, evaluate, validate as validate_mapping)

CS = [.001, .01, .1, 1.]
COMPONENTS = [10, 25, 50, 100]


def project(pack, n_components):
    x = pack['train']
    assert n_components < min(x.shape), (n_components, x.shape)
    pca = PCA(n_components=n_components, svd_solver='full', whiten=False)
    pca.fit(x)
    assert pca.singular_values_[-1] > 1e-10
    a = pca.transform(x)
    b = (pack['test'] - pca.mean_) @ pca.components_.T
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(n_components), atol=1e-10)
    return dict(train=a, test=b, pca=pca)


def tune(data, train, splits, components=None, cs=None):
    components = COMPONENTS if components is None else components
    cs = CS if cs is None else cs
    keys = [(k, c) for k in [0]+components for c in cs]
    scores = {key: [] for key in keys}
    probabilities = {key: np.full(len(train), np.nan) for key in keys}
    seen = np.zeros(len(train), dtype=int)
    for a, b in splits:
        assert not np.intersect1d(a, b).size
        seen[b] += 1
        pack = data.pack(train[a], train[b])
        pc = project(pack, max(components))
        for k, c in keys:
            x, v = (pack['train'], pack['test']) if k == 0 else (pc['train'][:, :k], pc['test'][:, :k])
            model = fit_model(x, data.labels[train[a]], c, 'ridge', 0.)
            p = model.predict_proba(v)[:, 1]
            assert np.isfinite(p).all()
            probabilities[(k, c)][b] = p
            scores[(k, c)].append(float(roc_auc_score(data.labels[train[b]], p)))
    assert (seen == 1).all()
    rows = [dict(variant='original_ridge' if k == 0 else 'pca_ridge', n_components=k, C=c,
                 mean_inner_AUROC=float(np.mean(v)), inner_AUROCs=';'.join(map(str, v))) for (k, c), v in scores.items()]
    best = {}
    for variant in ['original_ridge', 'pca_ridge']:
        chosen = min([r for r in rows if r['variant'] == variant], key=lambda r: (-r['mean_inner_AUROC'], r['n_components'], r['C'])).copy()
        threshold, ba = select_threshold(data.labels[train], probabilities[(chosen['n_components'], chosen['C'])])
        chosen.update(threshold=threshold, inner_threshold_balanced_accuracy=ba)
        best[variant] = chosen
    return best, rows


def artifact(model, pack, pc, chosen, genes):
    k = chosen['n_components']
    loadings = pc['pca'].components_[:k].copy()
    weights = model.coef_[0] @ loadings
    return dict(classifier=model, hyperparameters=chosen, threshold=chosen['threshold'],
                feature_universe=np.asarray(genes), rna=pack['state'], selected_gene_ids=np.asarray(genes)[pack['state']['indices']],
                pca_mean=pc['pca'].mean_.copy(), pca_components=loadings,
                explained_variance_ratio=pc['pca'].explained_variance_ratio_[:k].copy(),
                equivalent_gene_coefficients=weights,
                equivalent_intercept=float(model.intercept_[0] - pc['pca'].mean_ @ weights), positive_class='PD')


def predict_artifact(state, raw):
    raw = np.asarray(raw)
    assert raw.ndim == 2 and raw.shape[1] == len(state['feature_universe'])
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw == np.floor(raw)).all()
    library = raw.sum(1, dtype=np.float64)
    assert (library > 0).all()
    rna = state['rna']
    x = np.log2(1 + raw[:, rna['indices']] / library[:, None] * 1e6)
    x = (x - rna['means']) / rna['scales']
    projected = (x - state['pca_mean']) @ state['pca_components'].T
    p = state['classifier'].predict_proba(projected)[:, 1]
    np.testing.assert_allclose(p, expit(x @ state['equivalent_gene_coefficients'] + state['equivalent_intercept']), atol=1e-9)
    return p


def validate(device):
    checks = validate_mapping(device)
    rng = np.random.default_rng(SEED)
    raw = rng.poisson(30, (80, 150)).astype(np.int32)
    y = np.tile([0, 1], 40)
    raw[y == 1, :5] += 12
    genes, indices = np.arange(150).astype(str), np.arange(120)
    tr, te = np.arange(60), np.arange(60, 80)
    data = MappedData(raw, genes, y, indices, device)
    splits = make_splits(y[tr], np.arange(60), 'participant_stratified', 3, SEED)
    best, rows = tune(data, tr, splits, components=[5, 10], cs=[.001, .1])
    altered, labels = raw.copy(), y.copy()
    altered[te] += 10000; labels[te] = 1 - labels[te]
    changed = MappedData(altered, genes, labels, indices, device)
    other, other_rows = tune(changed, tr, splits, components=[5, 10], cs=[.001, .1])
    assert best == other and rows == other_rows
    pack = data.pack(tr, te)
    pc = project(pack, 10)
    reverse = MappedData(raw, genes, 1-y, indices, device)
    np.testing.assert_allclose(pc['pca'].components_, project(reverse.pack(tr, te), 10)['pca'].components_, atol=1e-12)
    np.testing.assert_allclose(pc['test'], pc['pca'].transform(pack['test']), atol=1e-12)
    chosen = best['pca_ridge']; k = chosen['n_components']
    model = fit_model(pc['train'][:, :k], y[tr], chosen['C'], 'ridge', 0.)
    state = artifact(model, pack, pc, chosen, genes)
    with tempfile.TemporaryDirectory(prefix='pca_ridge_check_') as folder:
        path = Path(folder)/'model.joblib'
        joblib.dump(state, path)
        np.testing.assert_allclose(predict_artifact(joblib.load(path), raw[te]), model.predict_proba(pc['test'][:, :k])[:, 1], atol=1e-9)
    checks.update(PCA_holdout_tuning_threshold='PASS', PCA_label_independence='PASS',
                  PCA_orthonormality_transform='PASS', PCA_serialization_collapsed_inference='PASS')
    return checks
