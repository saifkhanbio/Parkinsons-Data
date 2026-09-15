"""Validate and launch the controlled mapped-gene RBF-SVM comparison."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
from mapped_svm import (BASE, PRIOR, HALLMARK, MAPPED, SVM, MappedData, choose_device,
                        make_splits, SEED, fit, tune, artifact, predict_artifact, validate)

ROOT = Path(__file__).resolve().parent
KEYS = ['repeat', 'protocol', 'fold']


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def status(state, **kwargs):
    save('status.json', dict(status=state, utc=datetime.now(timezone.utc).isoformat(), **kwargs))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def table(name, rows):
    pd.DataFrame(rows).to_csv(ROOT / name, sep='\t', index=False)


def prepare():
    status('PREPARING')
    old = json.loads((MAPPED / 'input_sha256.json').read_text())
    assert json.loads((MAPPED / 'summary.json').read_text())['status'] == 'COMPLETE'
    for path, expected in old.items():
        assert sha(path) == expected, path
    raw, genes = np.load(BASE / 'counts.npy'), np.load(BASE / 'gene_ids.npy')
    meta = pd.read_csv(BASE / 'metadata.tsv', sep='\t', dtype={'PATNO': str})
    assert raw.shape == (528, 58780) and len(genes) == 58780
    assert len(meta) == 528 and meta.PATNO.is_unique
    assert np.issubdtype(raw.dtype, np.integer) and (raw >= 0).all()
    assert meta.group.value_counts().to_dict() == {'PD': 358, 'Control': 170}
    assert np.array_equal(meta.label, meta.group.eq('PD').astype(int))
    mapping = np.load(HALLMARK / 'membership.npz')
    indices = mapping['indices']
    assert len(indices) == len(np.unique(indices)) == 4376
    candidates = pd.read_csv(MAPPED / 'candidate_genes.tsv', sep='\t')
    assert np.array_equal(candidates.raw_index, indices)
    assert np.array_equal(candidates.Geneid, genes[indices])
    table('candidate_genes.tsv', candidates)
    plan = json.loads((MAPPED / 'fold_plan.json').read_text())
    assert len(plan) == 30
    y, batches = meta.label.to_numpy(), meta.batch.to_numpy(str)
    coverage = {}
    for split in plan:
        tr, te = np.array(split['train_indices']), np.array(split['test_indices'])
        assert len(np.unique(tr)) == len(tr) and len(np.unique(te)) == len(te)
        assert not np.intersect1d(tr, te).size and set(np.r_[tr, te]) == set(range(528))
        assert len(np.unique(y[tr])) == len(np.unique(y[te])) == 2
        key = (split['repeat'], split['protocol'])
        coverage.setdefault(key, np.zeros(528, dtype=int))[te] += 1
        if split['protocol'] == 'batch_grouped':
            assert not set(batches[tr]) & set(batches[te])
        seen = np.zeros(len(tr), dtype=int)
        for inner in split['inner']:
            a, b = np.array(inner['train_positions']), np.array(inner['validation_positions'])
            seen[b] += 1
            assert not np.intersect1d(a, b).size and set(np.r_[a, b]) == set(range(len(tr)))
            assert len(np.unique(y[tr[a]])) == len(np.unique(y[tr[b]])) == 2
            if split['protocol'] == 'batch_grouped':
                assert not set(batches[tr[a]]) & set(batches[tr[b]])
        assert (seen == 1).all()
    assert len(coverage) == 6 and all((v == 1).all() for v in coverage.values())
    save('fold_plan.json', plan)
    device = choose_device()
    checks = validate(device)
    data = MappedData(raw, genes, y, indices, device)
    tr = np.array(plan[0]['train_indices'])
    inner = plan[0]['inner'][0]
    pack = data.pack(tr[np.array(inner['train_positions'])], tr[np.array(inner['validation_positions'])])
    checked = dict(status='PASS', participants=528, PD=358, Control=170, candidate_genes=4376,
                   first_inner_eligible_genes=pack['train'].shape[1], outer_folds=30, checks=checks,
                   device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   sklearn=sklearn.__version__, python=sys.executable)
    save('preflight.json', checked)
    paths = [Path(p) for p in old]
    paths += [MAPPED / n for n in ['mapped_genes.py', 'outer_gene_coefficients.tsv', 'outer_fold_metrics.tsv',
                                  'out_of_fold_predictions.tsv', 'summary.json', 'input_sha256.json']]
    paths += [SVM / 'svm_analysis.py']
    paths += [ROOT / n for n in ['README.md', 'mapped_svm.py', 'launch.py', 'predict.py', 'candidate_genes.tsv', 'fold_plan.json']]
    save('input_sha256.json', {str(p): sha(p) for p in paths})
    return checked


def summarize(metrics):
    current = pd.DataFrame(metrics)
    baseline = pd.read_csv(MAPPED / 'outer_fold_metrics.tsv', sep='\t')
    old_predictions = pd.read_csv(MAPPED / 'out_of_fold_predictions.tsv', sep='\t', dtype={'PATNO': str})
    for key, group in old_predictions.groupby(KEYS + ['penalty']):
        row = baseline.set_index(KEYS + ['penalty']).loc[key]
        np.testing.assert_allclose(roc_auc_score(group.label, group.probability_PD), row.test_AUROC, atol=1e-12)
    paired = baseline.merge(current, on=KEYS, suffixes=('_baseline', '_SVM'), validate='many_to_one')
    assert len(paired) == 60
    for name in ['n_train', 'n_test', 'eligible_genes']:
        assert np.array_equal(paired[name + '_baseline'], paired[name + '_SVM'])
    paired['AUROC_difference'] = paired.test_AUROC_SVM - paired.test_AUROC_baseline
    table('paired_fold_comparisons.tsv', paired)
    paired_repeat = paired.groupby(['repeat', 'protocol', 'penalty']).AUROC_difference.mean().reset_index()
    table('paired_repeat_comparisons.tsv', paired_repeat)
    table('paired_summary.tsv', paired_repeat.groupby(['protocol', 'penalty']).AUROC_difference.agg(['mean', 'min', 'max']).reset_index())
    combined = pd.concat([baseline.rename(columns={'penalty': 'algorithm'}), current.assign(algorithm='rbf_svm')], ignore_index=True)
    repeats = combined.groupby(['repeat', 'protocol', 'algorithm']).agg(
        AUROC=('test_AUROC', 'mean'), train_AUROC=('train_AUROC', 'mean'), eligible_genes=('eligible_genes', 'mean')).reset_index()
    repeats['train_test_gap'] = repeats.train_AUROC - repeats.AUROC
    table('primary_per_repeat.tsv', repeats)
    summary = repeats.groupby(['protocol', 'algorithm']).agg(AUROC_mean=('AUROC', 'mean'), AUROC_min=('AUROC', 'min'),
        AUROC_max=('AUROC', 'max'), train_AUROC_mean=('train_AUROC', 'mean'), train_test_gap_mean=('train_test_gap', 'mean')).reset_index()
    table('primary_summary.tsv', summary)
    return summary, repeats


def train():
    raw, genes = np.load(BASE / 'counts.npy'), np.load(BASE / 'gene_ids.npy')
    indices = np.load(HALLMARK / 'membership.npz')['indices']
    meta = pd.read_csv(BASE / 'metadata.tsv', sep='\t', dtype={'PATNO': str})
    y = meta.label.to_numpy()
    data = MappedData(raw, genes, y, indices, choose_device())
    plan = json.loads((ROOT / 'fold_plan.json').read_text())
    old = pd.read_csv(MAPPED / 'outer_gene_coefficients.tsv', sep='\t', usecols=KEYS + ['penalty', 'Geneid'])
    prior_features = {key: group.Geneid.to_numpy() for key, group in old[old.penalty == 'ridge'].groupby(KEYS)}
    assert len(prior_features) == 30
    old_predictions = pd.read_csv(MAPPED / 'out_of_fold_predictions.tsv', sep='\t', dtype={'PATNO': str})
    prior_tests = {key: group.set_index('PATNO').label.sort_index() for key, group in old_predictions[old_predictions.penalty == 'ridge'].groupby(KEYS)}
    predictions, metrics, params, tuning, features = [], [], [], [], []
    for number, split in enumerate(plan, 1):
        context = {k: split[k] for k in KEYS}
        status('TRAINING', completed_folds=number - 1, total_folds=30, **context)
        print(f'{datetime.now(timezone.utc).isoformat()} {number}/30 {context}', flush=True)
        tr, te = np.array(split['train_indices']), np.array(split['test_indices'])
        pd.testing.assert_series_equal(meta.iloc[te].set_index('PATNO').label.sort_index(), prior_tests[tuple(context.values())])
        inner = [(np.array(s['train_positions']), np.array(s['validation_positions'])) for s in split['inner']]
        chosen, rows = tune(data, tr, inner)
        tuning.extend(dict(**context, **r) for r in rows)
        pack = data.pack(tr, te)
        selected = genes[pack['state']['indices']]
        assert np.array_equal(selected, prior_features[tuple(context.values())])
        model = fit(pack['train'], y[tr], chosen['C'], chosen['gamma_multiplier'])
        margin = model.decision_function(pack['test'])
        native = (margin >= 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], native, labels=[0, 1]).ravel()
        predictions.extend(dict(**context, PATNO=meta.PATNO.iloc[i], label=int(y[i]), decision_margin=float(m),
                                predicted_PD_native=int(m >= 0)) for i, m in zip(te, margin))
        metrics.append(dict(**context, n_train=len(tr), n_test=len(te), eligible_genes=len(selected),
                            support_vectors=int(model.n_support_.sum()),
                            train_AUROC=float(roc_auc_score(y[tr], model.decision_function(pack['train']))),
                            test_AUROC=float(roc_auc_score(y[te], margin)),
                            native_balanced_accuracy=float(balanced_accuracy_score(y[te], native)),
                            native_sensitivity=float(tp / (tp + fn)), native_specificity=float(tn / (tn + fp))))
        params.append(dict(**context, **chosen, actual_gamma=float(model._gamma), eligible_genes=len(selected),
                           support_vectors=int(model.n_support_.sum()), optimizer_iterations=int(model.n_iter_[0])))
        state = artifact(model, pack, chosen, genes)
        np.testing.assert_allclose(predict_artifact(state, raw[te[:3]]), margin[:3], atol=1e-9, rtol=1e-9)
        features.extend(dict(**context, Geneid=g) for g in selected)
        for name, rows in [('out_of_fold_predictions.tsv', predictions), ('outer_fold_metrics.tsv', metrics),
                           ('selected_parameters.tsv', params), ('inner_tuning_scores.tsv', tuning)]:
            table(name, rows)
    pred = pd.DataFrame(predictions)
    assert len(pred) == 528 * 3 * 2 and not pred.duplicated(['repeat', 'protocol', 'PATNO']).any()
    table('outer_eligible_genes.tsv', features)
    status('SUMMARIZING')
    summary, repeats = summarize(metrics)
    status('FITTING_FINAL_DEVELOPMENT_MODEL')
    splits = make_splits(y, meta.batch.to_numpy(str), 'participant_stratified', 5, SEED + 9901)
    save('final_tuning_folds.json', [dict(train_indices=a.tolist(), validation_indices=b.tolist()) for a, b in splits])
    chosen, rows = tune(data, np.arange(528), splits)
    table('final_tuning_scores.tsv', rows)
    full = data.pack(np.arange(528), np.array([], dtype=int))
    model = fit(full['train'], y, chosen['C'], chosen['gamma_multiplier'])
    state = artifact(model, full, chosen, genes)
    state.update(training_n=528, candidate_gene_ids=genes[indices], sklearn_version=sklearn.__version__, numpy_version=np.__version__)
    (ROOT / 'models').mkdir(exist_ok=True)
    path = ROOT / 'models' / 'mapped_RNA_rbf_svm.joblib'
    joblib.dump(state, path)
    np.testing.assert_allclose(predict_artifact(joblib.load(path), raw[:5]), model.decision_function(full['train'][:5]), atol=1e-9)
    save('final_parameters.json', chosen)
    table('final_eligible_genes.tsv', dict(Geneid=state['selected_gene_ids']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    for i, (protocol, algorithm) in enumerate((p, a) for p in ['participant_stratified', 'batch_grouped'] for a in ['ridge', 'elasticnet', 'rbf_svm']):
        values = repeats[(repeats.protocol == protocol) & (repeats.algorithm == algorithm)].sort_values('repeat').AUROC
        assert len(values) == 3
        ax.scatter(i + np.array([-.06, 0, .06]), values)
        labels.append(f'{protocol}\n{algorithm}')
    ax.axhline(.5, color='gray', ls='--')
    ax.set(xticks=np.arange(6), xticklabels=labels, ylim=(0, 1), ylabel='Mean within-fold AUROC per repeat')
    ax.tick_params(axis='x', labelsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / 'mapped_svm_comparison.png', dpi=170)
    plt.close(fig)
    hashes = json.loads((ROOT / 'input_sha256.json').read_text())
    assert all(sha(p) == v for p, v in hashes.items())
    lines = ['# Controlled mapped-gene RBF-SVM comparison', '',
             '528 participants; fixed 4,376-gene candidate universe. Identical training-only preprocessing and 30 nested outer folds to the preserved mapped-gene ridge/elastic-net fits.', '',
             '| Protocol | Algorithm | Held-out AUROC (repeat range) | Training AUROC | Train–test gap |',
             '| --- | --- | --- | ---: | ---: |']
    for r in summary.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.algorithm} | {r.AUROC_mean:.3f} ({r.AUROC_min:.3f}–{r.AUROC_max:.3f}) | {r.train_AUROC_mean:.3f} | {r.train_test_gap_mean:.3f} |')
    lines += ['', 'Primary AUROC averages within-fold AUROCs, then three repeat means. Repeat ranges are descriptive, not confidence intervals. Paired SVM-minus-baseline differences are saved in paired_summary.tsv. No headline AUROC pools margins from separate SVMs.', '',
              'Native zero-margin classification metrics are secondary and do not use the optimized probability thresholds of the earlier logistic models. Margins are uncalibrated and are not probabilities.', '',
              'Cohort/count integrity, nested boundaries, identical eligible genes and test participants in all 30 comparisons, source hashes, CPU/GPU agreement, held-out perturbation, nonlinear synthetic behavior, convergence and serialized inference checks passed. One full-data SVM development artifact is saved.', '',
              'This is an adaptive exploratory comparison on the same cohort. Independent-cohort validation remains outstanding; internal performance does not establish clinical utility.', '',
              '![Comparison](mapped_svm_comparison.png)', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    save('summary.json', dict(status='COMPLETE', participants=528, candidate_genes=4376,
                             final_eligible_genes=full['train'].shape[1], primary_results=summary.to_dict(orient='records'),
                             validation='PASS', source_hashes_unchanged=True, external_validation=False))
    status('COMPLETE', report='RESULTS.md')


if __name__ == '__main__':
    if '--run' not in sys.argv and (ROOT / 'status.json').exists():
        raise SystemExit('Existing run; refusing duplicate launch')
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            if '--run' in sys.argv:
                train()
            else:
                checked = prepare()
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLCONFIGDIR=str(ROOT / 'matplotlib_cache'))
                (ROOT / 'matplotlib_cache').mkdir(exist_ok=True)
                status('LAUNCHING')
                with (ROOT / 'run.log').open('ab', buffering=0) as log:
                    job = subprocess.Popen([sys.executable, '-u', str(ROOT / 'launch.py'), '--run'], stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT.parent.parent, env=env, start_new_session=True)
                save('job.json', dict(pid=job.pid, device=checked['device_name'], log=str(ROOT / 'run.log')))
                print(json.dumps(dict(launched_pid=job.pid, preflight=checked), indent=2))
    except Exception:
        status('FAILED', error=traceback.format_exc())
        raise
