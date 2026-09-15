"""Validate and detach a reproducible classifier refinement job."""
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
from refine import (BASE, FAMILIES, PENALTIES, SEED, FoldData, choose_device,
                    make_splits, matrices, fit_model, tune_models, artifact,
                    predict_artifact, evaluate, validate_refinement, CLINICAL_FEATURES)

ROOT = Path(__file__).resolve().parent
PROTOCOLS = ['participant_stratified', 'batch_grouped']
SEEDS = [SEED + 10000 * i for i in range(3)]


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def status(state, **kw):
    save('status.json', dict(status=state, utc=datetime.now(timezone.utc).isoformat(), **kw))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare():
    status('PREPARING')
    assert json.loads((ROOT / 'solver_repair_validation.json').read_text())['status'] == 'PASS'
    original_hashes = json.loads((BASE / 'input_sha256.json').read_text())
    for name in ['counts.npy', 'gene_ids.npy', 'metadata.tsv', 'classifier.py', 'fold_plan.json']:
        path = BASE / name
        assert sha(path) == original_hashes[str(path)], f'Original input changed: {name}'
    meta = pd.read_csv(BASE / 'metadata.tsv', sep='\t', dtype={'PATNO': str})
    raw, genes = np.load(BASE / 'counts.npy', mmap_mode='r'), np.load(BASE / 'gene_ids.npy')
    assert len(meta) == 528 and meta.PATNO.is_unique
    assert meta.group.value_counts().to_dict() == {'PD': 358, 'Control': 170}
    assert np.array_equal(meta.label.to_numpy(), meta.group.eq('PD').astype(int).to_numpy())
    assert raw.shape == (528, 58780) and len(genes) == len(np.unique(genes)) == 58780
    assert np.issubdtype(raw.dtype, np.integer) and (raw >= 0).all() and (raw.sum(1) > 0).all()
    assert meta.batch.notna().all()
    y, batches = meta.label.to_numpy(), meta.batch.to_numpy(str)
    old_plan = json.loads((BASE / 'fold_plan.json').read_text())
    plan, assignments = [], []
    for repeat, seed in enumerate(SEEDS, 1):
        for protocol in PROTOCOLS:
            for fold, (tr, te) in enumerate(make_splits(y, batches, protocol, 5, seed), 1):
                inner = make_splits(y[tr], batches[tr], protocol, 3, seed + 100 * fold)
                split = dict(fold=fold, train_indices=tr.tolist(), test_indices=te.tolist(),
                             inner=[dict(train_positions=a.tolist(), validation_positions=b.tolist()) for a, b in inner])
                if repeat == 1:
                    assert split == old_plan[protocol][fold - 1]
                plan.append(dict(repeat=repeat, seed=seed, protocol=protocol, **split))
                assignments.extend(dict(repeat=repeat, protocol=protocol, fold=fold,
                                        PATNO=meta.PATNO.iloc[i], group=meta.group.iloc[i], batch=batches[i]) for i in te)
    save('fold_plan.json', plan)
    pd.DataFrame(assignments).to_csv(ROOT / 'outer_fold_assignments.tsv', sep='\t', index=False)
    device = choose_device()
    checked = validate_refinement(device)
    result = dict(status='PASS', participants=528, PD=358, Control=170, genes=58780,
                  repeats=3, outer_folds_total=len(plan), inner_folds=3, models_per_fold=6,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU fallback',
                  sklearn=sklearn.__version__, torch=torch.__version__, python=sys.executable, checks=checked)
    save('preflight.json', result)
    paths = [BASE / name for name in ['counts.npy', 'gene_ids.npy', 'metadata.tsv', 'classifier.py',
                                     'fold_plan.json', 'out_of_fold_predictions.tsv', 'input_sha256.json']]
    paths += [ROOT / name for name in ['README.md', 'refine.py', 'launch.py', 'predict.py', 'elastic_solver.py',
                                     'validate_solver_repair.py', 'solver_repair_validation.json', 'fold_plan.json']]
    save('input_sha256.json', {str(p): sha(p) for p in paths})
    return result


def summarize(pred):
    rows = []
    for (repeat, protocol, family, penalty), group in pred.groupby(['repeat', 'protocol', 'family', 'penalty']):
        group = group.sort_values('sample_index')
        assert group.sample_index.tolist() == list(range(528))
        y, p = group.label.to_numpy(), group.probability_PD.to_numpy()
        for mode, threshold in [('fixed_0_5', .5), ('inner_selected', group.threshold.to_numpy())]:
            rows.append(dict(repeat=int(repeat), protocol=protocol, family=family, penalty=penalty,
                             threshold_mode=mode, **evaluate(y, p, threshold)))
    performance = pd.DataFrame(rows)
    performance.to_csv(ROOT / 'performance_by_repeat.tsv', sep='\t', index=False)
    keys = ['protocol', 'family', 'penalty', 'threshold_mode']
    metrics = ['AUROC', 'average_precision', 'Brier', 'balanced_accuracy', 'sensitivity', 'specificity']
    stability = performance.groupby(keys)[metrics].agg(['mean', 'min', 'max'])
    stability.columns = ['_'.join(c) for c in stability.columns]
    stability = stability.reset_index()
    stability.to_csv(ROOT / 'stability_summary.tsv', sep='\t', index=False)
    paired = []
    for (repeat, protocol, mode), group in performance.groupby(['repeat', 'protocol', 'threshold_mode']):
        indexed = group.set_index(['family', 'penalty'])
        pairs = [(f'{f}: elasticnet minus ridge', (f, 'elasticnet'), (f, 'ridge')) for f in FAMILIES]
        pairs += [(f'{penalty}: combined minus blood_clinical', ('combined', penalty), ('blood_clinical', penalty)) for penalty in PENALTIES]
        for label, a, b in pairs:
            paired.append(dict(repeat=int(repeat), protocol=protocol, threshold_mode=mode, comparison=label,
                               **{m + '_difference': float(indexed.loc[a, m] - indexed.loc[b, m]) for m in metrics}))
    pd.DataFrame(paired).to_csv(ROOT / 'paired_comparisons.tsv', sep='\t', index=False)
    # The same held-out participants and ridge models must reproduce the preserved benchmark.
    original = pd.read_csv(BASE / 'out_of_fold_predictions.tsv', sep='\t', dtype={'PATNO': str})
    original = original[original.family.isin(FAMILIES)]
    repeated = pred[(pred.repeat == 1) & (pred.penalty == 'ridge')]
    merged = repeated.merge(original, on=['protocol', 'family', 'PATNO'], validate='one_to_one', suffixes=('_new', '_old'))
    assert len(merged) == 528 * 3 * 2
    delta = float(np.max(np.abs(merged.probability_PD_new - merged.probability_PD_old)))
    np.testing.assert_allclose(merged.probability_PD_new, merged.probability_PD_old, atol=1e-8, rtol=1e-8)
    save('benchmark_reproduction.json', dict(status='PASS', maximum_probability_difference=delta))
    return stability, paired


def train():
    torch.set_num_threads(2)
    meta = pd.read_csv(BASE / 'metadata.tsv', sep='\t', dtype={'PATNO': str})
    raw, genes = np.load(BASE / 'counts.npy'), np.load(BASE / 'gene_ids.npy')
    y, batches = meta.label.to_numpy(), meta.batch.to_numpy(str)
    data = FoldData(raw, genes, meta, y, choose_device())
    plan = json.loads((ROOT / 'fold_plan.json').read_text())
    predictions, parameters, tuning, features = [], [], [], []
    for number, split in enumerate(plan, 1):
        context = {key: split[key] for key in ['repeat', 'protocol', 'fold']}
        status('TRAINING_NESTED_CV', outer_jobs_completed=number - 1, outer_jobs_total=len(plan), **context)
        print(f'{datetime.now(timezone.utc).isoformat()} {number}/{len(plan)} {context}', flush=True)
        tr, te = np.array(split['train_indices']), np.array(split['test_indices'])
        inner = [(np.array(s['train_positions']), np.array(s['validation_positions'])) for s in split['inner']]
        best, table = tune_models(data, tr, inner)
        tuning.extend(dict(**context, **r) for r in table)
        pack = data.pack(tr, te)
        for (family, penalty), chosen in best.items():
            x, v = matrices(pack, family, chosen['k'])
            model = fit_model(x, y[tr], chosen['C'], penalty, chosen['l1_ratio'])
            p = model.predict_proba(v)[:, 1]
            assert np.isfinite(p).all()
            threshold = chosen['threshold']
            predictions.extend(dict(**context, family=family, penalty=penalty, sample_index=int(i),
                                    PATNO=meta.PATNO.iloc[i], label=int(y[i]), probability_PD=float(prob),
                                    threshold=threshold, predicted_PD=bool(prob >= threshold)) for i, prob in zip(te, p))
            parameters.append(dict(**context, **chosen, total_features=x.shape[1], nonzero_coefficients=int(np.count_nonzero(model.coef_)),
                                   elasticnet_kkt_residual=float(model.kkt_residual_) if penalty == 'elasticnet' else None,
                                   optimizer_iterations=int(getattr(model, 'total_optimizer_iterations_', model.n_iter_[0]))))
            state = artifact(model, pack, family, chosen['k'], genes)
            state.update(threshold=threshold, penalty=penalty)
            np.testing.assert_allclose(predict_artifact(state, raw[te[:3]], meta.iloc[te[:3]]), p[:3], atol=1e-9, rtol=1e-9)
            if family in ['rna', 'combined']:
                offset = len(CLINICAL_FEATURES['blood_clinical']) if family == 'combined' else 0
                for gene, coef in zip(state['selected_gene_ids'], model.coef_[0, offset:]):
                    features.append(dict(**context, family=family, penalty=penalty, Geneid=gene,
                                         coefficient=float(coef), nonzero=bool(coef != 0)))
        pd.DataFrame(predictions).to_csv(ROOT / 'out_of_fold_predictions.tsv', sep='\t', index=False)
        pd.DataFrame(parameters).to_csv(ROOT / 'outer_selected_parameters.tsv', sep='\t', index=False)
        pd.DataFrame(tuning).to_csv(ROOT / 'inner_tuning_scores.tsv', sep='\t', index=False)
    pred = pd.DataFrame(predictions)
    assert len(pred) == 528 * 3 * 2 * 6
    assert not pred.duplicated(['repeat', 'protocol', 'family', 'penalty', 'PATNO']).any()
    feature_table = pd.DataFrame(features)
    feature_table.to_csv(ROOT / 'outer_gene_coefficients.tsv', sep='\t', index=False)
    frequency = feature_table.groupby(['protocol', 'family', 'penalty', 'Geneid']).agg(
        folds_screened=('coefficient', 'size'), folds_nonzero=('nonzero', 'sum'), mean_coefficient_when_screened=('coefficient', 'mean'))
    frequency.to_csv(ROOT / 'gene_stability.tsv', sep='\t')
    status('SUMMARIZING')
    stability, paired = summarize(pred)
    status('FITTING_FINAL_DEVELOPMENT_MODELS')
    splits = make_splits(y, batches, 'participant_stratified', 5, SEED + 9901)
    save('final_tuning_folds.json', [dict(train_indices=a.tolist(), validation_indices=b.tolist()) for a, b in splits])
    best, table = tune_models(data, np.arange(528), splits)
    pd.DataFrame(table).to_csv(ROOT / 'final_tuning_scores.tsv', sep='\t', index=False)
    full = data.pack(np.arange(528), np.array([], dtype=int))
    (ROOT / 'models').mkdir(exist_ok=True)
    for (family, penalty), chosen in best.items():
        x, _ = matrices(full, family, chosen['k'])
        model = fit_model(x, y, chosen['C'], penalty, chosen['l1_ratio'])
        state = artifact(model, full, family, chosen['k'], genes)
        state.update(threshold=chosen['threshold'], penalty=penalty, hyperparameters=chosen,
                     training_n=528, sklearn_version=sklearn.__version__, numpy_version=np.__version__)
        path = ROOT / 'models' / f'{family}_{penalty}.joblib'
        joblib.dump(state, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(predict_artifact(restored, raw[:5], meta.iloc[:5]), model.predict_proba(x[:5])[:, 1], atol=1e-9)
        assert restored['threshold'] == chosen['threshold']
    save('final_parameters.json', list(best.values()))
    hashes = json.loads((ROOT / 'input_sha256.json').read_text())
    assert all(sha(path) == expected for path, expected in hashes.items()), 'Source input changed during analysis'
    # Repeats are shown individually to avoid presenting their range as a confidence interval.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    performance = pd.read_csv(ROOT / 'performance_by_repeat.tsv', sep='\t')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [f'{f}\n{p}' for f in FAMILIES for p in PENALTIES]
    for row, protocol in enumerate(PROTOCOLS):
        for col, metric in enumerate(['AUROC', 'balanced_accuracy']):
            for i, (family, penalty) in enumerate((f, p) for f in FAMILIES for p in PENALTIES):
                part = performance[(performance.protocol == protocol) & (performance.family == family) &
                                   (performance.penalty == penalty) & (performance.threshold_mode == 'inner_selected')]
                axes[row, col].scatter(np.arange(3) * .06 + i - .06, part.sort_values('repeat')[metric])
            axes[row, col].set(xticks=np.arange(6), xticklabels=labels, ylim=(.35, .85),
                               title=protocol, ylabel=metric)
            axes[row, col].axhline(.5, color='gray', ls='--')
            axes[row, col].tick_params(axis='x', labelsize=7)
    fig.tight_layout(); fig.savefig(ROOT / 'refinement_stability.png', dpi=160); plt.close(fig)
    rows = ['# PPMI classifier refinement', '',
            '528 participants (358 PD, 170 controls); three repeats of nested five-fold evaluation. This refinement followed inspection of the initial results and remains exploratory internal validation.', '',
            'Values below are repeat means, with AUROC min–max across repeats. This range is not a confidence interval. Sensitivity/specificity and balanced accuracy use thresholds selected only inside each outer training set.', '',
            '| Protocol | Family | Penalty | AUROC mean (range) | Balanced accuracy | Sensitivity | Specificity |',
            '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for r in stability[stability.threshold_mode == 'inner_selected'].itertuples(index=False):
        rows.append(f'| {r.protocol} | {r.family} | {r.penalty} | {r.AUROC_mean:.3f} ({r.AUROC_min:.3f}–{r.AUROC_max:.3f}) | {r.balanced_accuracy_mean:.3f} | {r.sensitivity_mean:.3f} | {r.specificity_mean:.3f} |')
    rows += ['', 'Threshold changes do not alter AUROC. Fixed-0.5 results, AP, Brier, and paired model comparisons are saved in the TSV tables. Three repeats reuse the same participants; they are not independent evidence. No external validation, clinical utility, or population calibration has been established.', '',
             'Repeat-1 ridge probabilities reproduced the original benchmark. All fold, preprocessing, threshold leakage, convergence, inference roundtrip, and source-integrity checks passed. Six final full-cohort development models contain fitted transforms and training-selected thresholds for future evaluation.', '',
             '![Repeated-split results](refinement_stability.png)', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(rows))
    save('summary.json', dict(status='COMPLETE', participants=528, repeats=3, external_validation=False,
                             source_hashes_unchanged=True, benchmark_reproduction='PASS', artifact_roundtrips='PASS',
                             stability=stability.to_dict(orient='records'), paired_comparisons=paired))
    status('COMPLETE', report='RESULTS.md')


if __name__ == '__main__':
    if '--run' not in sys.argv and (ROOT / 'status.json').exists():
        raise SystemExit('Run already exists; refusing a duplicate launch.')
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            if '--run' in sys.argv:
                train()
            else:
                checked = prepare()
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           MPLCONFIGDIR=str(ROOT / 'matplotlib_cache'))
                (ROOT / 'matplotlib_cache').mkdir(exist_ok=True)
                status('LAUNCHING')
                with (ROOT / 'run.log').open('ab', buffering=0) as log:
                    job = subprocess.Popen([sys.executable, '-u', str(ROOT / 'launch.py'), '--run'],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, cwd=ROOT.parent.parent, start_new_session=True)
                save('job.json', dict(pid=job.pid, device=checked['device_name'], log=str(ROOT / 'run.log')))
                print(json.dumps(dict(launched_pid=job.pid, preflight=checked), indent=2))
    except Exception:
        status('FAILED', error=traceback.format_exc())
        raise
