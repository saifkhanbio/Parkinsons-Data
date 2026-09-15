"""Training-fold RNA selection and regularized PD classification utilities."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             accuracy_score, confusion_matrix, brier_score_loss)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

FAMILIES = ['demographic', 'blood_clinical', 'technical', 'rna', 'combined']
CS = [0.001, 0.01, 0.1, 1.0]
KS = [100, 500, 2000]
SEED = 20260912
CLINICAL_FEATURES = {
    'demographic': ['age_collection_years', 'male'],
    'blood_clinical': ['age_collection_years', 'male', 'log_wbc', 'neutrophils_percent',
                       'monocytes_percent', 'eosinophils_percent', 'basophils_percent'],
    'technical': ['RIN', 'intergenic_percent', 'log_library', 'phase2'],
}


def choose_device():
    if torch.cuda.is_available():
        choices = [i for i in range(torch.cuda.device_count()) if 'RTX' in torch.cuda.get_device_name(i)]
        return torch.device('cuda', choices[0] if choices else 0)
    return torch.device('cpu')


def clinical_matrix(metadata, family, libraries):
    m = metadata.copy()
    if family in ['demographic', 'blood_clinical']:
        assert m.sex.isin(['Female', 'Male']).all(), 'Unrecognized sex category'
        m['male'] = m.sex.eq('Male').astype(float)
    if family == 'blood_clinical':
        assert (m.wbc > 0).all()
        m['log_wbc'] = np.log(m.wbc)
    if family == 'technical':
        assert m.phase.isin(['PPMI-Phase1', 'PPMI-Phase2']).all(), 'Unrecognized sequencing phase'
        m['phase2'] = m.phase.eq('PPMI-Phase2').astype(float)
        m['log_library'] = np.log(libraries)
    x = m[CLINICAL_FEATURES[family]].to_numpy(dtype=float)
    assert np.isfinite(x).all(), 'Missing or nonfinite required covariates'
    return x


def make_splits(y, batches, protocol, n_splits, seed):
    if protocol == 'participant_stratified':
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
    else:
        assert protocol == 'batch_grouped'
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, batches))
    seen = np.zeros(len(y), dtype=int)
    for train, test in splits:
        assert not np.intersect1d(train, test).size
        assert len(np.unique(y[train])) == len(np.unique(y[test])) == 2
        if protocol == 'batch_grouped':
            assert not set(batches[train]) & set(batches[test])
        seen[test] += 1
    assert (seen == 1).all()
    return splits


class FoldData:
    def __init__(self, counts, genes, metadata, labels, device):
        self.raw = np.asarray(counts)
        self.genes = np.asarray(genes)
        self.metadata = metadata
        self.labels = np.asarray(labels)
        self.device = device
        assert self.raw.shape == (len(labels), len(genes))
        assert np.isfinite(self.raw).all() and (self.raw >= 0).all()
        self.libraries = self.raw.sum(axis=1, dtype=np.int64)
        assert (self.libraries > 0).all()
        self.raw_gpu = torch.as_tensor(self.raw, device=device)
        libs = torch.as_tensor(self.libraries, device=device, dtype=torch.float64)
        self.rna_gpu = torch.log2(1 + self.raw_gpu.to(torch.float64) / libs[:, None] * 1e6)
        # CPU reference for the sample-local transform, before learned preprocessing.
        np.testing.assert_allclose(self.rna_gpu[:8, :512].cpu().numpy(),
                                   np.log2(1 + self.raw[:8, :512] / self.libraries[:8, None] * 1e6),
                                   atol=1e-12, rtol=1e-12)
        self.clinical = {family: clinical_matrix(metadata, family, self.libraries) for family in CLINICAL_FEATURES}
        self.checked_cpu = False

    def pack(self, train, test):
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        assert not np.intersect1d(train, test).size
        x = self.rna_gpu[torch.as_tensor(train, device=self.device)]
        y = self.labels[train]
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        assert n0 > 1 and n1 > 1
        mean, variance = x.mean(0), x.var(0, unbiased=False)
        detected = (self.raw_gpu[torch.as_tensor(train, device=self.device)] >= 10).sum(0)
        eligible = (detected >= int(np.ceil(.2 * len(train)))) & (variance > 1e-24)
        x0 = x[torch.as_tensor(y == 0, device=self.device)]
        x1 = x[torch.as_tensor(y == 1, device=self.device)]
        pooled = (x0.var(0, unbiased=True) * (n0 - 1) + x1.var(0, unbiased=True) * (n1 - 1)) / (n0 + n1 - 2)
        score = (x1.mean(0) - x0.mean(0)).square() / (pooled.clamp(min=1e-20) * (1 / n0 + 1 / n1))
        score_cpu = score.cpu().numpy()
        choices = np.flatnonzero(eligible.cpu().numpy())
        assert len(choices) >= 100, 'Too few eligible genes'
        ordered = choices[np.lexsort((choices, -score_cpu[choices]))]
        selected = ordered[:min(max(KS), len(ordered))]
        means = mean[selected].cpu().numpy()
        scales = variance[selected].sqrt().cpu().numpy()
        tx = ((x[:, selected] - mean[selected]) / variance[selected].sqrt()).cpu().numpy()
        vx = ((self.rna_gpu[torch.as_tensor(test, device=self.device)][:, selected] - mean[selected]) / variance[selected].sqrt()).cpu().numpy()
        if not self.checked_cpu:
            # Independently recompute all top features' ANOVA scores/statistics on CPU.
            cpu = np.log2(1 + self.raw[train][:, selected] / self.libraries[train, None] * 1e6)
            a, b = cpu[y == 0], cpu[y == 1]
            pv = (a.var(0, ddof=1) * (n0 - 1) + b.var(0, ddof=1) * (n1 - 1)) / (n0 + n1 - 2)
            fs = (b.mean(0) - a.mean(0)) ** 2 / (np.maximum(pv, 1e-20) * (1 / n0 + 1 / n1))
            np.testing.assert_allclose(score_cpu[selected], fs, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(means, cpu.mean(0), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(scales, cpu.std(0), rtol=1e-10, atol=1e-10)
            self.checked_cpu = True
        packed = {'rna_train': tx, 'rna_test': vx, 'rna_state': {'indices': selected, 'means': means, 'scales': scales},
                  'eligible_genes': len(choices), 'clinical': {}}
        for family, values in self.clinical.items():
            mu, sd = values[train].mean(0), values[train].std(0)
            sd[sd < 1e-12] = 1
            packed['clinical'][family] = {'train': (values[train] - mu) / sd, 'test': (values[test] - mu) / sd,
                                           'means': mu, 'scales': sd, 'features': CLINICAL_FEATURES[family]}
        return packed


def matrices(pack, family, k):
    if family in CLINICAL_FEATURES:
        v = pack['clinical'][family]
        return v['train'], v['test']
    a, b = pack['rna_train'][:, :k], pack['rna_test'][:, :k]
    if family == 'combined':
        c = pack['clinical']['blood_clinical']
        a, b = np.column_stack([c['train'], a]), np.column_stack([c['test'], b])
    return a, b


def fit_logistic(x, y, c):
    for iterations in [2000, 10000]:
        model = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=iterations, tol=1e-7,
                                   class_weight=None, random_state=SEED)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y)
        bad = any(issubclass(item.category, ConvergenceWarning) for item in messages)
        if not bad:
            assert model.classes_.tolist() == [0, 1]
            assert np.isfinite(model.coef_).all()
            return model
    raise RuntimeError('Logistic optimizer failed to converge after retry')


def tune(data, train, inner_splits):
    # Every cache is specific to an inner training partition, never an outer/full-data fit.
    scores = {(family, k, c): [] for family in FAMILIES for k in (KS if family in ['rna', 'combined'] else [0]) for c in CS}
    for inner_train, inner_test in inner_splits:
        tr, va = train[inner_train], train[inner_test]
        pack = data.pack(tr, va)
        for family, k, c in scores:
            x, v = matrices(pack, family, k)
            model = fit_logistic(x, data.labels[tr], c)
            scores[(family, k, c)].append(float(roc_auc_score(data.labels[va], model.predict_proba(v)[:, 1])))
    table = [{'family': f, 'k': k, 'C': c, 'mean_inner_AUROC': float(np.mean(s)), 'inner_AUROCs': ';'.join(map(str, s))}
             for (f, k, c), s in scores.items()]
    best = {}
    for family in FAMILIES:
        rows = [row for row in table if row['family'] == family]
        best[family] = sorted(rows, key=lambda r: (-r['mean_inner_AUROC'], r['k'], r['C']))[0]
    return best, table


def artifact(model, pack, family, k, genes):
    clinical = 'blood_clinical' if family == 'combined' else family
    state = {'family': family, 'classifier': model, 'feature_universe': np.asarray(genes),
             'transform': 'log2(1 + sample_local_CPM)', 'positive_class': 'PD', 'threshold': .5}
    if family in ['rna', 'combined']:
        state['rna'] = {key: value[:k].copy() for key, value in pack['rna_state'].items()}
        state['selected_gene_ids'] = np.asarray(genes)[state['rna']['indices']]
    if clinical in CLINICAL_FEATURES:
        state['clinical'] = {key: pack['clinical'][clinical][key] for key in ['means', 'scales', 'features']}
        state['clinical_family'] = clinical
    return state


def predict_artifact(state, raw, metadata):
    raw = np.asarray(raw)
    assert raw.ndim == 2 and raw.shape[1] == len(state['feature_universe'])
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw == np.floor(raw)).all()
    libraries = raw.sum(axis=1, dtype=np.float64)
    assert (libraries > 0).all()
    features = []
    if 'clinical' in state:
        s = state['clinical']
        values = clinical_matrix(metadata, state['clinical_family'], libraries)
        features.append((values - s['means']) / s['scales'])
    if 'rna' in state:
        s = state['rna']
        values = np.log2(1 + raw[:, s['indices']] / libraries[:, None] * 1e6)
        features.append((values - s['means']) / s['scales'])
    x = np.column_stack(features)
    assert np.isfinite(x).all()
    return state['classifier'].predict_proba(x)[:, 1]


def metrics(y, probabilities):
    pred = probabilities >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'AUROC': float(roc_auc_score(y, probabilities)), 'average_precision': float(average_precision_score(y, probabilities)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'accuracy': float(accuracy_score(y, pred)),
            'sensitivity': float(tp / (tp + fn)), 'specificity': float(tn / (tn + fp)),
            'Brier': float(brier_score_loss(y, probabilities)), 'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}


def validate_leakage(device):
    rng = np.random.default_rng(SEED)
    raw = rng.poisson(30, size=(80, 300)).astype(np.int32)
    y = np.tile([0, 1], 40)
    meta = pd.DataFrame({'age_collection_years': rng.normal(60, 8, 80), 'sex': np.where(y, 'Male', 'Female'),
                         'wbc': rng.uniform(4, 8, 80), 'neutrophils_percent': rng.uniform(40, 70, 80),
                         'monocytes_percent': rng.uniform(3, 8, 80), 'eosinophils_percent': rng.uniform(1, 3, 80),
                         'basophils_percent': rng.uniform(.1, 1, 80), 'RIN': rng.uniform(6, 9, 80),
                         'intergenic_percent': rng.uniform(5, 15, 80), 'phase': 'PPMI-Phase1'})
    tr, te = np.arange(60), np.arange(60, 80)
    a = FoldData(raw, np.arange(300).astype(str), meta, y, device).pack(tr, te)
    changed, new_y, new_meta = raw.copy(), y.copy(), meta.copy()
    changed[te] += rng.integers(0, 100000, (20, 300)); new_y[te] = 1 - new_y[te]
    new_meta.loc[te, 'age_collection_years'] += 1000
    b = FoldData(changed, np.arange(300).astype(str), new_meta, new_y, device).pack(tr, te)
    for key in ['indices', 'means', 'scales']:
        np.testing.assert_array_equal(a['rna_state'][key], b['rna_state'][key])
    np.testing.assert_array_equal(a['rna_train'], b['rna_train'])
    for family in CLINICAL_FEATURES:
        np.testing.assert_array_equal(a['clinical'][family]['train'], b['clinical'][family]['train'])
    model = fit_logistic(a['rna_train'][:, :100], y[tr], .01)
    state = artifact(model, a, 'rna', 100, np.arange(300).astype(str))
    np.testing.assert_allclose(predict_artifact(state, raw[te], meta.iloc[te]), model.predict_proba(a['rna_test'][:, :100])[:, 1], atol=1e-10)
    return {'held_out_label_count_covariate_perturbation': 'Training selection/statistics unchanged',
            'CPU_GPU_feature_statistics': 'PASS', 'inference_roundtrip': 'PASS'}
