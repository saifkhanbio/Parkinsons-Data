"""Prepare and launch a self-contained nested classifier evaluation without polling."""
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
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import calibration_curve

from classifier import (FAMILIES, CLINICAL_FEATURES, SEED, FoldData, choose_device, make_splits,
                        validate_leakage, tune, matrices, fit_logistic, artifact, predict_artifact, metrics)

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PROTOCOLS = ['participant_stratified', 'batch_grouped']


def save(name, value):
    p = ROOT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def status(state, **kwargs):
    save('status.json', {'status': state, 'utc': datetime.now(timezone.utc).isoformat(), **kwargs})


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    assert not (ROOT / 'status.json').exists(), 'This directory already contains a run; do not launch twice.'
    status('PREPARING')
    manifest = WORK / 'ppmi-blood-counts-2026-09-12/approved_cohort_528/cohort_manifest.tsv'
    covariates = WORK / 'ppmi-deseq2-2026-09-12/primary_colData.tsv'
    counts_path = WORK / 'ppmi-expression-qc-2026-09-12/raw_counts.tsv.gz'
    cohort = pd.read_csv(manifest, sep='\t', dtype={'PATNO': str})
    meta = pd.read_csv(covariates, sep='\t', dtype={'PATNO': str}).set_index('PATNO').loc[cohort.PATNO].reset_index()
    assert len(cohort) == 528 and cohort.PATNO.is_unique
    assert cohort.group.value_counts().to_dict() == {'PD': 358, 'Control': 170}
    assert meta.PATNO.tolist() == cohort.PATNO.tolist() and meta.group.tolist() == cohort.group.tolist()
    for col in ['wbc', 'neutrophils_percent', 'lymphocytes_percent', 'monocytes_percent', 'eosinophils_percent', 'basophils_percent']:
        meta[col] = cohort[col].to_numpy(float)
    meta['CBC_month_gap'] = cohort.month_gap_lab_minus_rna.to_numpy(int)
    assert meta.CBC_month_gap.isin([-2, -1, 0]).all()
    meta['label'] = meta.group.eq('PD').astype(int)
    meta.to_csv(ROOT / 'metadata.tsv', sep='\t', index=False)
    raw = pd.read_csv(counts_path, sep='\t', usecols=['Geneid'] + cohort.PATNO.tolist())
    assert len(raw) == 58780 and raw.Geneid.is_unique
    x = raw[cohort.PATNO].to_numpy().T
    assert np.isfinite(x).all() and (x >= 0).all() and (x == np.floor(x)).all()
    assert x.max() < np.iinfo(np.int32).max and (x.sum(1) > 0).all()
    np.save(ROOT / 'counts.npy', x.astype(np.int32))
    np.save(ROOT / 'gene_ids.npy', raw.Geneid.to_numpy(dtype=str))
    del raw, x
    y, batches = meta.label.to_numpy(), meta.batch.to_numpy(str)
    plan, assignments = {}, []
    for protocol in PROTOCOLS:
        plan[protocol] = []
        for fold, (tr, te) in enumerate(make_splits(y, batches, protocol, 5, SEED), 1):
            inner = make_splits(y[tr], batches[tr], protocol, 3, SEED + 100 * fold)
            plan[protocol].append({'fold': fold, 'train_indices': tr.tolist(), 'test_indices': te.tolist(),
                                   'inner': [{'train_positions': a.tolist(), 'validation_positions': b.tolist()} for a, b in inner]})
            assignments += [{'protocol': protocol, 'outer_test_fold': fold, 'PATNO': meta.PATNO.iloc[i],
                             'group': meta.group.iloc[i], 'batch': meta.batch.iloc[i]} for i in te]
    save('fold_plan.json', plan)
    pd.DataFrame(assignments).to_csv(ROOT / 'outer_fold_assignments.tsv', sep='\t', index=False)
    dev = choose_device()
    checks = validate_leakage(dev)
    save('preflight.json', {'status': 'PASS', 'participants': 528, 'PD': 358, 'Control': 170, 'raw_genes': 58780,
                            'batches': int(meta.batch.nunique()), 'device': str(dev),
                            'device_name': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'CPU fallback',
                            'python': sys.executable, 'sklearn': sklearn.__version__, 'torch': torch.__version__,
                            'leakage_and_inference_checks': checks, 'outer_folds_per_protocol': 5, 'inner_folds': 3})
    inputs = [manifest, covariates, counts_path, ROOT / 'classifier.py', ROOT / 'launch.py', ROOT / 'predict.py',
              ROOT / 'README.md', ROOT / 'counts.npy', ROOT / 'gene_ids.npy', ROOT / 'metadata.tsv', ROOT / 'fold_plan.json']
    save('input_sha256.json', {str(p): sha(p) for p in inputs})
    return json.loads((ROOT / 'preflight.json').read_text())


def bootstrap_intervals(frame, batches, protocol):
    wide = frame.pivot(index='sample_index', columns='family', values='probability_PD').sort_index()
    ys = frame.drop_duplicates('sample_index').set_index('sample_index').loc[wide.index, 'label'].to_numpy()
    probabilities = wide[FAMILIES].to_numpy()
    rng = np.random.default_rng(SEED + (protocol == 'batch_grouped'))
    buckets = {family: [] for family in FAMILIES}; deltas = []
    group_values = batches[wide.index.to_numpy()]
    groups = np.unique(group_values)
    by_group = {g: np.flatnonzero(group_values == g) for g in groups}
    for _ in range(1000):
        if protocol == 'batch_grouped':
            ix = np.concatenate([by_group[g] for g in rng.choice(groups, len(groups), replace=True)])
        else:
            ix = np.concatenate([rng.choice(np.flatnonzero(ys == k), int((ys == k).sum()), replace=True) for k in [0, 1]])
        if len(np.unique(ys[ix])) != 2:
            continue
        aucs = {}
        for j, family in enumerate(FAMILIES):
            p = probabilities[ix, j]
            auc = float(roc_auc_score(ys[ix], p)); aucs[family] = auc
            buckets[family].append([auc, average_precision_score(ys[ix], p), balanced_accuracy_score(ys[ix], p >= .5)])
        deltas.append(aucs['combined'] - aucs['blood_clinical'])
    intervals = []
    for family, values in buckets.items():
        bounds = np.quantile(values, [.025, .975], axis=0)
        record = {'protocol': protocol, 'family': family, 'bootstrap_replicates': len(values)}
        for j, metric in enumerate(['AUROC', 'average_precision', 'balanced_accuracy']):
            record[metric + '_CI_low'], record[metric + '_CI_high'] = map(float, bounds[:, j])
        intervals.append(record)
    return intervals, {'protocol': protocol, 'comparison': 'combined minus blood_clinical AUROC',
                       'difference': float(roc_auc_score(ys, wide.combined) - roc_auc_score(ys, wide.blood_clinical)),
                       'CI_low': float(np.quantile(deltas, .025)), 'CI_high': float(np.quantile(deltas, .975)),
                       'method': 'conditional batch-cluster bootstrap' if protocol == 'batch_grouped' else 'conditional stratified participant bootstrap'}


def train():
    meta = pd.read_csv(ROOT / 'metadata.tsv', sep='\t', dtype={'PATNO': str})
    raw, genes = np.load(ROOT / 'counts.npy'), np.load(ROOT / 'gene_ids.npy')
    y, batches = meta.label.to_numpy(), meta.batch.to_numpy(str)
    data = FoldData(raw, genes, meta, y, choose_device())
    plan = json.loads((ROOT / 'fold_plan.json').read_text())
    predictions, fold_scores, parameters, tuning_rows, features = [], [], [], [], []
    for protocol in PROTOCOLS:
        for item in plan[protocol]:
            fold = item['fold']; tr = np.array(item['train_indices']); te = np.array(item['test_indices'])
            status('TRAINING_NESTED_CV', protocol=protocol, outer_fold=fold, outer_folds=5)
            print(f'{datetime.now(timezone.utc).isoformat()} {protocol} outer fold {fold}', flush=True)
            inner = [(np.array(s['train_positions']), np.array(s['validation_positions'])) for s in item['inner']]
            best, rows = tune(data, tr, inner)
            tuning_rows += [{'protocol': protocol, 'outer_fold': fold, **r} for r in rows]
            pack = data.pack(tr, te)
            for family in FAMILIES:
                chosen = best[family]; k, c = chosen['k'], chosen['C']
                x, v = matrices(pack, family, k)
                model = fit_logistic(x, y[tr], c)
                p = model.predict_proba(v)[:, 1]
                assert len(p) == len(te) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                predictions += [{'protocol': protocol, 'outer_fold': fold, 'family': family, 'sample_index': int(i),
                                 'PATNO': meta.PATNO.iloc[i], 'label': int(y[i]), 'probability_PD': float(prob),
                                 'predicted_PD_threshold_0_5': bool(prob >= .5)} for i, prob in zip(te, p)]
                fold_scores.append({'protocol': protocol, 'outer_fold': fold, 'family': family, 'n_test': len(te), **metrics(y[te], p)})
                parameters.append({'protocol': protocol, 'outer_fold': fold, **chosen, 'eligible_training_genes': pack['eligible_genes'],
                                   'actual_features': x.shape[1], 'optimizer_iterations': int(model.n_iter_[0]), 'intercept': float(model.intercept_[0])})
                if family in ['rna', 'combined']:
                    state = pack['rna_state']; n = min(k, len(state['indices']))
                    offset = len(CLINICAL_FEATURES['blood_clinical']) if family == 'combined' else 0
                    for j in range(n):
                        features.append({'protocol': protocol, 'outer_fold': fold, 'family': family,
                                         'Geneid': genes[state['indices'][j]], 'standardized_coefficient': float(model.coef_[0, offset + j]),
                                         'training_mean_log_CPM': float(state['means'][j]), 'training_scale_log_CPM': float(state['scales'][j])})
                # Verify inference on held-out raw data uses exactly the saved training transform.
                state = artifact(model, pack, family, k, genes)
                np.testing.assert_allclose(predict_artifact(state, raw[te[:3]], meta.iloc[te[:3]]), p[:3], atol=1e-9, rtol=1e-9)
            pd.DataFrame(predictions).to_csv(ROOT / 'out_of_fold_predictions.tsv', sep='\t', index=False)
            pd.DataFrame(parameters).to_csv(ROOT / 'outer_selected_parameters.tsv', sep='\t', index=False)
    pred = pd.DataFrame(predictions)
    assert len(pred) == len(meta) * len(FAMILIES) * len(PROTOCOLS)
    assert not pred.duplicated(['protocol', 'family', 'PATNO']).any()
    results, intervals, increments = [], [], []
    status('SUMMARIZING_NESTED_CV')
    for protocol in PROTOCOLS:
        part = pred[pred.protocol == protocol]
        for family in FAMILIES:
            r = part[part.family == family].sort_values('sample_index')
            assert r.sample_index.tolist() == list(range(528))
            results.append({'protocol': protocol, 'family': family, 'n': len(r), 'PD_prevalence': float(r.label.mean()),
                            **metrics(r.label.to_numpy(), r.probability_PD.to_numpy())})
        ci, difference = bootstrap_intervals(part, batches, protocol)
        intervals.extend(ci); increments.append(difference)
    results = pd.DataFrame(results).merge(pd.DataFrame(intervals), on=['protocol', 'family'], validate='one_to_one')
    results.to_csv(ROOT / 'performance.tsv', sep='\t', index=False)
    pd.DataFrame(fold_scores).to_csv(ROOT / 'outer_fold_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(tuning_rows).to_csv(ROOT / 'inner_tuning_scores.tsv', sep='\t', index=False)
    pd.DataFrame(increments).to_csv(ROOT / 'RNA_increment_over_blood_baseline.tsv', sep='\t', index=False)
    features = pd.DataFrame(features)
    features.to_csv(ROOT / 'outer_selected_gene_coefficients.tsv', sep='\t', index=False)
    frequency = features.groupby(['protocol', 'family', 'Geneid']).agg(outer_folds_selected=('outer_fold', 'nunique'),
                    mean_standardized_coefficient=('standardized_coefficient', 'mean')).reset_index()
    frequency.sort_values(['protocol', 'family', 'outer_folds_selected'], ascending=[True, True, False]).to_csv(ROOT / 'gene_selection_frequency.tsv', sep='\t', index=False)
    status('FITTING_FINAL_DEVELOPMENT_ARTIFACTS')
    final_splits = make_splits(y, batches, 'participant_stratified', 5, SEED + 9901)
    save('final_tuning_folds.json', [{'train_indices': tr.tolist(), 'validation_indices': te.tolist()} for tr, te in final_splits])
    best, full_tuning = tune(data, np.arange(528), final_splits)
    pd.DataFrame(full_tuning).to_csv(ROOT / 'final_development_tuning.tsv', sep='\t', index=False)
    full = data.pack(np.arange(528), np.array([], dtype=int))
    (ROOT / 'models').mkdir(exist_ok=True)
    for family in FAMILIES:
        chosen = best[family]; x, _ = matrices(full, family, chosen['k'])
        model = fit_logistic(x, y, chosen['C'])
        state = artifact(model, full, family, chosen['k'], genes)
        state.update(training_n=528, hyperparameters=chosen, sklearn_version=sklearn.__version__, numpy_version=np.__version__)
        path = ROOT / 'models' / f'{family}.joblib'
        joblib.dump(state, path)
        reloaded = joblib.load(path)
        np.testing.assert_allclose(predict_artifact(reloaded, raw[:5], meta.iloc[:5]), model.predict_proba(x[:5])[:, 1], atol=1e-9, rtol=1e-9)
    save('final_development_parameters.json', best)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, protocol in enumerate(PROTOCOLS):
        for family in FAMILIES:
            r = pred[(pred.protocol == protocol) & (pred.family == family)]
            RocCurveDisplay.from_predictions(r.label, r.probability_PD, name=family, ax=axes[row, 0])
            PrecisionRecallDisplay.from_predictions(r.label, r.probability_PD, name=family, ax=axes[row, 1])
            observed, predicted = calibration_curve(r.label, r.probability_PD, n_bins=5, strategy='quantile')
            axes[row, 2].plot(predicted, observed, marker='o', label=family)
        axes[row, 0].plot([0, 1], [0, 1], '--', color='gray')
        axes[row, 1].axhline(y.mean(), ls='--', color='gray')
        axes[row, 2].plot([0, 1], [0, 1], '--', color='gray')
        axes[row, 2].set(xlabel='Mean predicted PD probability', ylabel='Observed PD fraction', xlim=(0, 1), ylim=(0, 1))
        for col, title in enumerate(['ROC', 'Precision–recall', 'Calibration']):
            axes[row, col].set_title(protocol + ': ' + title)
        axes[row, 2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / 'classifier_performance.png', dpi=170); plt.close(fig)
    summary = {'status': 'COMPLETE', 'n': 528, 'PD': 358, 'Control': 170, 'models': FAMILIES,
               'protocols': PROTOCOLS, 'performance': results.to_dict(orient='records'), 'RNA_increment': increments,
               'preflight': json.loads((ROOT / 'preflight.json').read_text()),
               'all_outer_samples_predicted_once_per_model_protocol': True, 'serialized_artifact_roundtrip': 'PASS',
               'external_validation': False, 'fixed_full_cohort_gene_shortlist_used': False}
    hashes = json.loads((ROOT / 'input_sha256.json').read_text())
    assert all(sha(Path(path)) == value for path, value in hashes.items()), 'Source input changed'
    summary['input_hashes_unchanged'] = True
    save('summary.json', summary)
    lines = ['# Initial PPMI classifier results', '', '528 participants: 358 PD, 170 controls. All results below use nested held-out predictions. PD prevalence is 67.8%; that is the no-skill average-precision reference.', '',
             '| Protocol | Model | AUROC (95% descriptive CI) | AP | Balanced accuracy | Sensitivity | Specificity | Brier |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.family} | {r.AUROC:.3f} ({r.AUROC_CI_low:.3f}–{r.AUROC_CI_high:.3f}) | {r.average_precision:.3f} | {r.balanced_accuracy:.3f} | {r.sensitivity:.3f} | {r.specificity:.3f} | {r.Brier:.3f} |')
    lines += ['', 'Threshold-based metrics use the prespecified probability threshold 0.5. Bootstrap intervals condition on fixed out-of-fold predictions and do not include retraining variability. Batch-grouped intervals resample whole batches.', '']
    for r in increments:
        lines.append(f"- {r['protocol']}: combined-minus-blood_clinical AUROC difference {r['difference']:.3f}, descriptive paired interval {r['CI_low']:.3f} to {r['CI_high']:.3f}.")
    lines += ['', 'All preprocessing, count filtering, supervised RNA selection and hyperparameter tuning were fitted within training partitions. No earlier significant-gene set or shortlist was used. Batch-grouped splits held phase–plate groups apart in both outer and inner folds. Final full-data models in models/ are development artifacts for future independent testing; their training/tuning scores are not independent performance estimates.', '',
              'The selected PPMI case/control population, prior full-cohort scientific analysis and QC, possible site/ancestry/technical effects, and screening counts predating RNA limit generalization. Internal classification does not establish disease-specific biomarkers, clinical utility, population calibration, or external replication. Choosing the strongest family after this comparison adds model-selection optimism.', '',
              'Detailed predictions, fold definitions, tuning scores, feature-selection frequencies, and executable inference artifacts are saved alongside this report. GPU numerical agreement, holdout-perturbation leakage checks, inference roundtrips and unchanged source hashes passed.', '',
              '![Performance plots](classifier_performance.png)', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    status('COMPLETE', report='RESULTS.md', summary='summary.json')


def run():
    try:
        with threadpool_limits(limits=2):
            train()
    except Exception:
        status('FAILED', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    if '--run' in sys.argv:
        run()
    else:
        try:
            with threadpool_limits(limits=2):
                checked = prepare()
            env = os.environ.copy()
            env.update(VIRTUAL_ENV=sys.prefix, PATH=str(Path(sys.executable).resolve().parent) + os.pathsep + env['PATH'],
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLCONFIGDIR=str(ROOT / 'matplotlib_cache'))
            (ROOT / 'matplotlib_cache').mkdir(exist_ok=True)
            status('LAUNCHING')
            with (ROOT / 'run.log').open('ab', buffering=0) as log:
                job = subprocess.Popen([sys.executable, '-u', str(ROOT / 'launch.py'), '--run'], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=WORK.parent)
            save('job.json', {'pid': job.pid, 'log': str(ROOT / 'run.log'), 'device': checked['device_name']})
            print(json.dumps({'launched_pid': job.pid, 'preflight': checked}, indent=2))
        except Exception:
            status('FAILED', error=traceback.format_exc())
            raise
