"""Frozen-split CatBoost comparison using training-only Hallmark pathway scores."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import os
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / 'matplotlib_cache')
WORK = ROOT.parent
HALL = WORK / 'ppmi-hallmark-classifier-2026-09-12'
BOOST = WORK / 'ppmi-mapped-gene-boosting-2026-09-14'
MAPPED = WORK / 'ppmi-mapped-gene-classifier-2026-09-12'
sys.path.insert(0, str(BOOST))
from estimators import fit, score, GRIDS
sys.path.insert(0, str(HALL))
from pathways import PathwayData, BASE, choose_device
import json
import hashlib
import time
import subprocess
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits
KEYS = ['repeat', 'protocol', 'fold']
GRID = GRIDS['catboost']


def save(name, obj):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def table(name, rows):
    pd.DataFrame(rows).to_csv(ROOT / name, sep='\t', index=False)


def status(state, **kw):
    save('status.json', dict(status=state, utc=datetime.now(timezone.utc).isoformat(), **kw))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data():
    raw = np.load(BASE / 'counts.npy')
    genes = np.load(BASE / 'gene_ids.npy')
    meta = pd.read_csv(BASE / 'metadata.tsv', sep='\t', dtype={'PATNO': str})
    mapping = np.load(HALL / 'membership.npz')
    device = choose_device()
    assert device.type == 'cuda', 'GPU required; no silent CPU fallback'
    data = PathwayData(raw, genes, meta, meta.label.to_numpy(), mapping['indices'],
                       mapping['membership'], mapping['names'], device)
    return raw, genes, meta, data


def predict(state, raw):
    raw = np.asarray(raw)
    assert raw.ndim == 2 and raw.shape[1] == len(state['feature_universe'])
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw == np.floor(raw)).all()
    libs = raw.sum(1, dtype=np.float64)
    assert (libs > 0).all()
    s = state['pathway_state']
    x = np.log2(1 + raw[:, s['indices']] / libs[:, None] * 1e6)
    x = ((x - s['gene_means']) / s['gene_scales']) @ s['weights']
    x = (x - s['pathway_means']) / s['pathway_scales']
    return score(state['classifier'], x, 'catboost')


def artifact(model, pack, genes, chosen):
    return dict(classifier=model, algorithm='catboost', predictors='50 Hallmark pathway scores',
                pathway_state=pack['pathway_state'], feature_universe=genes,
                hyperparameters=chosen, threshold=.5, positive_class='PD',
                transform='sample-local full-universe logCPM; training gene z-scores; pathway means; training pathway z-scores')


def tune(data, train, splits, tag, completed):
    values = [[] for _ in GRID]
    seen = np.zeros(len(train), dtype=int)
    for j, (a, b) in enumerate(splits, 1):
        seen[b] += 1
        pack = data.pack(train[a], train[b], tag=f'{tag}_inner_{j}')
        for k, params in enumerate(GRID):
            status('TUNING', completed_folds=completed, total_folds=30,
                   partition=tag, inner_fold=j, candidate=k + 1, candidates=len(GRID))
            model = fit(pack['path_train'], data.labels[train[a]], 'catboost', params)
            values[k].append(float(roc_auc_score(data.labels[train[b]], score(model, pack['path_test'], 'catboost'))))
    assert (seen == 1).all()
    rows = [dict(candidate=i, parameters=GRID[i], mean_inner_AUROC=float(np.mean(v)),
                 inner_AUROCs=';'.join(map(str, v))) for i, v in enumerate(values)]
    return min(rows, key=lambda r: (-r['mean_inner_AUROC'], r['candidate'])), rows


def prepare():
    status('PREFLIGHT')
    hashes = {}
    for folder in [HALL, MAPPED, BOOST]:
        assert json.loads((folder / 'summary.json').read_text())['status'] == 'COMPLETE'
        for p, expected in json.loads((folder / 'input_sha256.json').read_text()).items():
            assert sha(p) == expected, p
            hashes[p] = expected
    raw, genes, meta, data = load_data()
    assert raw.shape == (528, 58780) and meta.PATNO.is_unique
    assert np.issubdtype(raw.dtype, np.integer) and (raw >= 0).all()
    assert meta.group.value_counts().to_dict() == {'PD': 358, 'Control': 170}
    assert np.array_equal(meta.label, meta.group.eq('PD').astype(int))
    assert len(data.indices) == len(np.unique(data.indices)) == 4376
    assert len(np.unique(data.names)) == 50
    plan = json.loads((HALL / 'fold_plan.json').read_text())
    assert plan == json.loads((MAPPED / 'fold_plan.json').read_text()) and len(plan) == 30
    counts = {}
    for split in plan:
        tr, te = np.array(split['train_indices']), np.array(split['test_indices'])
        assert len(np.unique(tr)) == len(tr) and len(np.unique(te)) == len(te)
        assert not np.intersect1d(tr, te).size and set(np.r_[tr, te]) == set(range(528))
        assert len(np.unique(data.labels[tr])) == len(np.unique(data.labels[te])) == 2
        counts.setdefault((split['repeat'], split['protocol']), np.zeros(528, int))[te] += 1
        seen = np.zeros(len(tr), int)
        if split['protocol'] == 'batch_grouped':
            assert not set(meta.batch.iloc[tr]) & set(meta.batch.iloc[te])
        for inner in split['inner']:
            a, b = np.array(inner['train_positions']), np.array(inner['validation_positions'])
            assert not np.intersect1d(a, b).size and set(np.r_[a, b]) == set(range(len(tr)))
            assert len(np.unique(data.labels[tr[a]])) == len(np.unique(data.labels[tr[b]])) == 2
            seen[b] += 1
            if split['protocol'] == 'batch_grouped':
                assert not set(meta.batch.iloc[tr[a]]) & set(meta.batch.iloc[tr[b]])
        assert (seen == 1).all()
    assert len(counts) == 6 and all((x == 1).all() for x in counts.values())
    save('fold_plan.json', plan)
    tr, te = np.array(plan[0]['train_indices']), np.array(plan[0]['test_indices'])
    pack = data.pack(tr, te)
    altered = raw.copy(); altered[te] += 10000
    labels = data.labels.copy(); labels[te] = 1 - labels[te]
    changed = PathwayData(altered, genes, meta, labels, data.indices, data.membership, data.names, data.device)
    for inner in plan[0]['inner']:
        a, b = np.array(inner['train_positions']), np.array(inner['validation_positions'])
        p, q = data.pack(tr[a], tr[b]), changed.pack(tr[a], tr[b])
        for key in ['path_train', 'path_test']:
            np.testing.assert_array_equal(p[key], q[key])
        np.testing.assert_array_equal(data.labels[tr[a]], changed.labels[tr[a]])
        np.testing.assert_array_equal(data.labels[tr[b]], changed.labels[tr[b]])
    # Clinical variables cannot enter the chosen predictor matrices.
    changed.clinical[:] = 0
    np.testing.assert_array_equal(pack['path_train'], changed.pack(tr, te)['path_train'])
    started = time.monotonic()
    model = fit(pack['path_train'], data.labels[tr], 'catboost', GRID[0])
    state = artifact(model, pack, genes, GRID[0])
    path = ROOT / 'preflight_model.joblib'
    joblib.dump(state, path)
    np.testing.assert_allclose(predict(joblib.load(path), raw[te]), score(model, pack['path_test'], 'catboost'), atol=1e-7)
    import catboost
    save('environment.json', dict(python=sys.version, executable=sys.executable, catboost=catboost.__version__,
                                numpy=np.__version__, torch=torch.__version__, device=torch.cuda.get_device_name(0)))
    checks = dict(status='PASS', split_identity=True, nested_boundaries=True, CPU_GPU_pathways=True,
                  heldout_training_input_invariance=True, clinical_not_predictors=True,
                  serialized_inference=True, gpu_fit_seconds=time.monotonic()-started)
    save('preflight.json', checks)
    for folder, names in [(HALL, ['pathways.py', 'membership.npz', 'outer_fold_metrics.tsv',
                               'out_of_fold_predictions.tsv', 'training_pathway_coverage.tsv', 'final_tuning_folds.json']),
                          (MAPPED, ['outer_fold_metrics.tsv']), (BOOST, ['estimators.py']),
                          (ROOT, ['run.py', 'predict.py', 'README.md', 'fold_plan.json', 'environment.json'])]:
        for name in names:
            p = folder / name; hashes[str(p)] = sha(p)
    save('input_sha256.json', hashes)
    return checks


def train():
    raw, genes, meta, data = load_data()
    plan = json.loads((ROOT / 'fold_plan.json').read_text())
    old_pred = pd.read_csv(HALL / 'out_of_fold_predictions.tsv', sep='\t', dtype={'PATNO': str})
    old_pred = old_pred[(old_pred.family == 'pathways') & (old_pred.penalty == 'ridge')]
    metrics, predictions, tuning_rows, params = [], [], [], []
    started = time.monotonic()
    for number, split in enumerate(plan, 1):
        context = {k: split[k] for k in KEYS}
        print(datetime.now(timezone.utc).isoformat(), number, '/30', context, flush=True)
        tr, te = np.array(split['train_indices']), np.array(split['test_indices'])
        reference = old_pred
        for key, val in context.items(): reference = reference[reference[key] == val]
        pd.testing.assert_series_equal(meta.iloc[te].set_index('PATNO').label.sort_index(), reference.set_index('PATNO').label.sort_index())
        inner = [(np.array(s['train_positions']), np.array(s['validation_positions'])) for s in split['inner']]
        chosen, rows = tune(data, tr, inner, f'outer_{number}', number - 1)
        tuning_rows.extend(dict(**context, **r) for r in rows)
        pack = data.pack(tr, te, tag=f'outer_{number}_fit')
        assert pack['path_train'].shape == (len(tr), 50)
        model = fit(pack['path_train'], data.labels[tr], 'catboost', chosen['parameters'])
        values = score(model, pack['path_test'], 'catboost')
        np.testing.assert_allclose(predict(artifact(model, pack, genes, chosen), raw[te]), values, atol=1e-7)
        tn, fp, fn, tp = confusion_matrix(data.labels[te], values >= .5, labels=[0, 1]).ravel()
        metrics.append(dict(**context, algorithm='pathway_catboost', n_train=len(tr), n_test=len(te),
              train_AUROC=float(roc_auc_score(data.labels[tr], score(model, pack['path_train'], 'catboost'))),
              test_AUROC=float(roc_auc_score(data.labels[te], values)), AP=float(average_precision_score(data.labels[te], values)),
              Brier=float(brier_score_loss(data.labels[te], values)), sensitivity=float(tp/(tp+fn)),
              specificity=float(tn/(tn+fp)), threshold=.5))
        predictions.extend(dict(**context, PATNO=meta.PATNO.iloc[i], label=int(data.labels[i]), probability=float(v)) for i, v in zip(te, values))
        params.append(dict(**context, **chosen))
        for name, rows in [('outer_fold_metrics.tsv', metrics), ('out_of_fold_predictions.tsv', predictions),
                           ('inner_tuning_scores.tsv', tuning_rows), ('selected_parameters.tsv', params),
                           ('training_pathway_coverage.tsv', data.coverage)]: table(name, rows)
        status('TRAINING', completed_folds=number, total_folds=30, elapsed_seconds=time.monotonic()-started)
    p = pd.DataFrame(predictions)
    assert len(p) == 528 * 6 and not p.duplicated(['repeat', 'protocol', 'PATNO']).any()
    # Exact member coverage comparison for all 90 inner and 30 outer partitions.
    prior = pd.read_csv(HALL / 'training_pathway_coverage.tsv', sep='\t')
    prior = prior[prior.partition.str.startswith('outer_')].sort_values(['partition', 'pathway']).reset_index(drop=True)
    current = pd.DataFrame(data.coverage).sort_values(['partition', 'pathway']).reset_index(drop=True)
    pd.testing.assert_frame_equal(current, prior, check_dtype=False)
    baseline = pd.read_csv(HALL / 'outer_fold_metrics.tsv', sep='\t')
    baseline = baseline[(baseline.family == 'pathways') & (baseline.penalty == 'ridge')].assign(algorithm='pathway_ridge')
    gene = pd.read_csv(MAPPED / 'outer_fold_metrics.tsv', sep='\t')
    gene = gene[gene.penalty == 'ridge'].assign(algorithm='gene_ridge')
    combined = pd.concat([pd.DataFrame(metrics), baseline, gene], ignore_index=True)
    repeats = combined.groupby(['repeat', 'protocol', 'algorithm']).agg(AUROC=('test_AUROC', 'mean'), train_AUROC=('train_AUROC', 'mean')).reset_index()
    repeats['train_test_gap'] = repeats.train_AUROC - repeats.AUROC
    summary = repeats.groupby(['protocol', 'algorithm']).agg(AUROC_mean=('AUROC', 'mean'), AUROC_min=('AUROC', 'min'), AUROC_max=('AUROC', 'max'), train_AUROC=('train_AUROC', 'mean')).reset_index()
    table('primary_per_repeat.tsv', repeats); table('primary_summary.tsv', summary)
    pairs = pd.DataFrame(metrics).merge(pd.concat([baseline, gene]), on=KEYS, suffixes=('_new', '_baseline'))
    assert len(pairs) == 60
    for col in ['n_train', 'n_test']: np.testing.assert_array_equal(pairs[col+'_new'], pairs[col+'_baseline'])
    pairs['AUROC_difference'] = pairs.test_AUROC_new - pairs.test_AUROC_baseline
    table('paired_fold_comparisons.tsv', pairs)
    table('paired_repeat_comparisons.tsv', pairs.groupby(['repeat', 'protocol', 'algorithm_baseline']).AUROC_difference.mean().reset_index())
    final_plan = json.loads((HALL / 'final_tuning_folds.json').read_text())
    save('final_tuning_folds.json', final_plan)
    final_splits = [(np.array(s['train_indices']), np.array(s['validation_indices'])) for s in final_plan]
    chosen, rows = tune(data, np.arange(528), final_splits, 'final', 30)
    table('final_tuning_scores.tsv', rows); save('final_parameters.json', chosen)
    pack = data.pack(np.arange(528), np.array([], dtype=int), tag='final_fit')
    model = fit(pack['path_train'], data.labels, 'catboost', chosen['parameters'])
    state = artifact(model, pack, genes, chosen)
    state.update(training_n=528, training_PATNO=meta.PATNO.to_numpy())
    (ROOT / 'models').mkdir(exist_ok=True)
    path = ROOT / 'models' / 'hallmark_pathway_catboost.joblib'
    joblib.dump(state, path)
    np.testing.assert_allclose(predict(joblib.load(path), raw), score(model, pack['path_train'], 'catboost'), atol=1e-7)
    table('training_pathway_coverage.tsv', data.coverage)
    table('final_pathway_importance.tsv', dict(pathway=data.names, prediction_value_change=model.feature_importances_))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    algorithms = ['pathway_ridge', 'pathway_catboost', 'gene_ridge']
    for ax, protocol in zip(axes, ['participant_stratified', 'batch_grouped']):
        for i, alg in enumerate(algorithms):
            vals = repeats[(repeats.protocol == protocol) & (repeats.algorithm == alg)].AUROC.to_numpy()
            ax.scatter(i + np.array([-.05, 0, .05]), vals)
        ax.axhline(.5, color='gray', ls='--')
        ax.set(xticks=range(3), xticklabels=['Pathway ridge', 'Pathway CatBoost', 'Gene ridge'], ylim=(.45, .75), title=protocol)
        ax.tick_params(axis='x', labelrotation=15)
    axes[0].set_ylabel('Mean within-fold AUROC per repeat')
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(ROOT / ('model_comparison.' + ext), dpi=600)
    plt.close(fig)
    hashes = json.loads((ROOT / 'input_sha256.json').read_text())
    assert all(sha(p) == h for p, h in hashes.items())
    lines = ['# CatBoost with 50 Hallmark pathway scores', '',
             '528 participants; identical 30 nested outer folds and training-only pathway scores. Ridge benchmarks are preserved.', '',
             '| Protocol | Model | Mean AUROC | Repeat range | Training AUROC |', '|---|---|---:|---|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.protocol} | {r.algorithm} | {r.AUROC_mean:.3f} | {r.AUROC_min:.3f}–{r.AUROC_max:.3f} | {r.train_AUROC:.3f} |')
    lines += ['', 'Repeat ranges are descriptive, not confidence intervals. Native .5 threshold metrics are secondary; probabilities are uncalibrated. All source-integrity, nested-split, matched-label, pathway-coverage and inference checks passed.', '',
              'This adaptive exploratory comparison uses an already examined cohort. Pathway scores summarize expression and are not direct pathway-activity measurements. Full-data feature importance is descriptive model reliance, not pathway significance or causal influence.', '',
              'One final CatBoost artifact, OOF predictions, tuning evidence, paired comparisons, and PNG/PDF figures are saved. The manuscript is unchanged.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    save('summary.json', dict(status='COMPLETE', participants=528, pathways=50, validation='PASS', source_hashes_unchanged=True, primary_results=summary.to_dict('records'), elapsed_seconds=time.monotonic()-started))
    status('COMPLETE', completed_folds=30, total_folds=30, report='RESULTS.md')


if __name__ == '__main__':
    if '--run' not in sys.argv and (ROOT / 'status.json').exists():
        raise SystemExit('Existing run: refusing duplicate launch')
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            if '--run' in sys.argv:
                train()
            else:
                checks = prepare()
                env = os.environ.copy()
                env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                with (ROOT / 'run.log').open('ab', buffering=0) as log:
                    job = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'run.py'), '--run'],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        cwd=ROOT.parent.parent, env=env, start_new_session=True)
                save('job.json', dict(pid=job.pid, device=torch.cuda.get_device_name(0)))
                print(json.dumps(dict(launched_pid=job.pid, preflight=checks), indent=2))
    except Exception:
        status('FAILED', error=traceback.format_exc())
        raise
