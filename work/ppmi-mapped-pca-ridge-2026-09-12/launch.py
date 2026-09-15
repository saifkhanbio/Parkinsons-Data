"""PCA plus ridge with a reproduced gene-space ridge benchmark."""
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
from pca_ridge import (BASE, PRIOR, HALLMARK, MAPPED, MappedData, choose_device, make_splits,
    SEED, CS, COMPONENTS, fit_model, tune, project, artifact, predict_artifact, evaluate, validate)

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
    pc=project(pack,max(COMPONENTS))
    assert pc['train'].shape[1]==100
    checked = dict(status='PASS', participants=528, PD=358, Control=170, candidate_genes=4376,
                   first_inner_eligible_genes=pack['train'].shape[1], outer_folds=30, checks=checks,
                   device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   sklearn=sklearn.__version__, python=sys.executable)
    save('preflight.json', checked)
    paths = [Path(p) for p in old]
    paths += [MAPPED / n for n in ['mapped_genes.py', 'outer_gene_coefficients.tsv', 'outer_fold_metrics.tsv',
                                  'out_of_fold_predictions.tsv', 'summary.json', 'input_sha256.json']]
    paths += [MAPPED / n for n in ['selected_parameters.tsv', 'inner_tuning_scores.tsv']]
    paths += [ROOT / n for n in ['README.md', 'pca_ridge.py', 'launch.py', 'predict.py', 'candidate_genes.tsv', 'fold_plan.json']]
    save('input_sha256.json', {str(p): sha(p) for p in paths})
    return checked



def summarize(pred, fold_metrics):
    pca = pd.DataFrame(fold_metrics).assign(variant='pca_ridge')
    old = pd.read_csv(MAPPED/'outer_fold_metrics.tsv', sep='\t')
    old = old[old.penalty=='ridge'].assign(variant='original_ridge')
    paired = pca.merge(old, on=KEYS, suffixes=('_pca', '_original'), validate='one_to_one')
    assert len(paired)==30
    for name in ['n_train','n_test','eligible_genes']:
        assert np.array_equal(paired[name+'_pca'],paired[name+'_original'])
    paired['AUROC_difference'] = paired.test_AUROC_pca-paired.test_AUROC_original
    table('paired_fold_comparisons.tsv',paired)
    per_repeat = paired.groupby(['repeat','protocol']).AUROC_difference.mean().reset_index()
    table('paired_repeat_comparisons.tsv',per_repeat)
    table('paired_summary.tsv',per_repeat.groupby('protocol').AUROC_difference.agg(['mean','min','max']).reset_index())
    combined = pd.concat([pca,old],ignore_index=True)
    repeats = combined.groupby(['repeat','protocol','variant']).agg(AUROC=('test_AUROC','mean'),
        train_AUROC=('train_AUROC','mean'),eligible_genes=('eligible_genes','mean')).reset_index()
    assert len(repeats)==12
    repeats['train_test_gap']=repeats.train_AUROC-repeats.AUROC
    table('primary_per_repeat.tsv',repeats)
    summary = repeats.groupby(['protocol','variant']).agg(AUROC_mean=('AUROC','mean'),AUROC_min=('AUROC','min'),
        AUROC_max=('AUROC','max'),train_AUROC_mean=('train_AUROC','mean'),train_test_gap_mean=('train_test_gap','mean')).reset_index()
    table('primary_summary.tsv',summary)
    secondary=[]
    for (repeat,protocol),group in pred.groupby(['repeat','protocol']):
        assert len(group)==528 and group.PATNO.is_unique
        for mode,threshold in [('fixed_0_5',.5),('inner_selected',group.threshold.to_numpy())]:
            secondary.append(dict(repeat=int(repeat),protocol=protocol,threshold_mode=mode,
                **evaluate(group.label.to_numpy(),group.probability_PD.to_numpy(),threshold)))
    table('secondary_pooled_OOF_metrics.tsv',secondary)
    return summary



def train():
    raw, genes = np.load(BASE/'counts.npy'), np.load(BASE/'gene_ids.npy')
    indices = np.load(HALLMARK/'membership.npz')['indices']
    meta = pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    y = meta.label.to_numpy()
    data = MappedData(raw,genes,y,indices,choose_device())
    plan = json.loads((ROOT/'fold_plan.json').read_text())
    old_params = pd.read_csv(MAPPED/'selected_parameters.tsv',sep='\t')
    old_params = old_params[old_params.penalty=='ridge'].set_index(KEYS)
    old_inner = pd.read_csv(MAPPED/'inner_tuning_scores.tsv',sep='\t')
    old_inner = old_inner[old_inner.penalty=='ridge'].set_index(KEYS+['C'])
    old_pred = pd.read_csv(MAPPED/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str})
    tests = {k:g.set_index('PATNO') for k,g in old_pred[old_pred.penalty=='ridge'].groupby(KEYS)}
    old_genes = pd.read_csv(MAPPED/'outer_gene_coefficients.tsv',sep='\t',usecols=KEYS+['penalty','Geneid'])
    feature_sets = {k:g.Geneid.to_numpy() for k,g in old_genes[old_genes.penalty=='ridge'].groupby(KEYS)}
    predictions, metrics, params, tuning, coefficients, checks = [], [], [], [], [], []
    for number, split in enumerate(plan,1):
        context = {k:split[k] for k in KEYS}; key=tuple(context.values())
        status('TRAINING',completed_folds=number-1,total_folds=30,**context)
        print(f'{datetime.now(timezone.utc).isoformat()} {number}/30 {context}',flush=True)
        tr, te = np.array(split['train_indices']), np.array(split['test_indices'])
        inner = [(np.array(s['train_positions']),np.array(s['validation_positions'])) for s in split['inner']]
        best, rows = tune(data,tr,inner)
        tuning.extend(dict(**context,**r) for r in rows)
        pack = data.pack(tr,te)
        assert np.array_equal(genes[pack['state']['indices']],feature_sets[key])
        previous = tests[key].loc[meta.PATNO.iloc[te]]
        assert len(previous)==len(te) and np.array_equal(previous.label,y[te])
        control = best['original_ridge']; old_chosen=old_params.loc[key]
        assert control['C']==old_chosen.C
        np.testing.assert_allclose(control['threshold'],old_chosen.threshold,atol=1e-10,rtol=1e-10)
        for row in rows:
            if row['variant']=='original_ridge':
                reference=old_inner.loc[key+(row['C'],)]
                np.testing.assert_allclose(row['mean_inner_AUROC'],reference.mean_inner_AUROC,atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.fromstring(row['inner_AUROCs'],sep=';'),np.fromstring(reference.inner_AUROCs,sep=';'),atol=1e-12,rtol=0)
        model = fit_model(pack['train'],y[tr],control['C'],'ridge',0.)
        control_p = model.predict_proba(pack['test'])[:,1]
        np.testing.assert_allclose(control_p,previous.probability_PD.to_numpy(),atol=1e-8,rtol=1e-8)
        checks.append(dict(**context,max_probability_difference=float(np.max(np.abs(control_p-previous.probability_PD.to_numpy()))),status='PASS'))
        chosen = best['pca_ridge']; k=chosen['n_components']
        pc = project(pack,max(COMPONENTS))
        model = fit_model(pc['train'][:,:k],y[tr],chosen['C'],'ridge',0.)
        p = model.predict_proba(pc['test'][:,:k])[:,1]
        state = artifact(model,pack,pc,chosen,genes)
        np.testing.assert_allclose(predict_artifact(state,raw[te[:3]]),p[:3],atol=1e-9,rtol=1e-9)
        predictions.extend(dict(**context,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(v),threshold=chosen['threshold']) for i,v in zip(te,p))
        variance = float(pc['pca'].explained_variance_ratio_[:k].sum())
        metrics.append(dict(**context,n_train=len(tr),n_test=len(te),eligible_genes=pack['train'].shape[1],n_components=k,
            explained_variance_ratio=variance,train_AUROC=float(roc_auc_score(y[tr],model.predict_proba(pc['train'][:,:k])[:,1])),
            test_AUROC=float(roc_auc_score(y[te],p))))
        params.append(dict(**context,**chosen,eligible_genes=pack['train'].shape[1],explained_variance_ratio=variance,
                           optimizer_iterations=int(model.n_iter_[0])))
        coefficients.extend(dict(**context,Geneid=g,equivalent_standardized_gene_coefficient=float(c))
                            for g,c in zip(state['selected_gene_ids'],state['equivalent_gene_coefficients']))
        for name, rows in [('out_of_fold_predictions.tsv',predictions),('outer_fold_metrics.tsv',metrics),
            ('selected_parameters.tsv',params),('inner_tuning_scores.tsv',tuning),('benchmark_reproduction.tsv',checks)]:
            table(name,rows)
    pred = pd.DataFrame(predictions)
    assert len(pred)==528*3*2 and not pred.duplicated(['repeat','protocol','PATNO']).any()
    table('outer_gene_coefficients.tsv',coefficients)
    status('SUMMARIZING')
    summary = summarize(pred,metrics)
    status('FITTING_FINAL_DEVELOPMENT_MODEL')
    splits = make_splits(y,meta.batch.to_numpy(str),'participant_stratified',5,SEED+9901)
    save('final_tuning_folds.json',[dict(train_indices=a.tolist(),validation_indices=b.tolist()) for a,b in splits])
    best, rows = tune(data,np.arange(528),splits)
    table('final_tuning_scores.tsv',rows)
    chosen = best['pca_ridge']; k=chosen['n_components']
    full = data.pack(np.arange(528),np.array([],dtype=int))
    pc = project(full,max(COMPONENTS))
    model = fit_model(pc['train'][:,:k],y,chosen['C'],'ridge',0.)
    state = artifact(model,full,pc,chosen,genes)
    state.update(training_n=528,candidate_gene_ids=genes[indices],sklearn_version=sklearn.__version__,numpy_version=np.__version__)
    (ROOT/'models').mkdir(exist_ok=True)
    path = ROOT/'models'/'mapped_PCA_ridge.joblib'
    joblib.dump(state,path)
    np.testing.assert_allclose(predict_artifact(joblib.load(path),raw[:5]),model.predict_proba(pc['train'][:5,:k])[:,1],atol=1e-9)
    save('final_parameters.json',chosen)
    table('final_PCA_explained_variance.tsv',dict(component=np.arange(1,k+1),explained_variance_ratio=state['explained_variance_ratio']))
    finalize(summary,full['train'].shape[1])


def finalize(summary,eligible_count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    repeats=pd.read_csv(ROOT/'primary_per_repeat.tsv',sep='\t')
    params=pd.read_csv(ROOT/'selected_parameters.tsv',sep='\t')
    table('selected_parameter_frequency.tsv',params.groupby(['protocol','n_components','C']).size().rename('folds').reset_index())
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,protocol in zip(axes,['participant_stratified','batch_grouped']):
        for repeat in sorted(repeats.repeat.unique()):
            values=repeats[(repeats.protocol==protocol)&(repeats.repeat==repeat)].set_index('variant').loc[['original_ridge','pca_ridge'],'AUROC']
            ax.plot([0,1],values,marker='o',label=f'Repeat {repeat}')
        ax.axhline(.5,color='gray',ls='--')
        ax.set(xticks=[0,1],xticklabels=['Original ridge','PCA + ridge'],ylim=(0,1),title=protocol)
    axes[0].set_ylabel('Mean within-fold AUROC per repeat'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT/'pca_ridge_comparison.png',dpi=170); plt.close(fig)
    hashes=json.loads((ROOT/'input_sha256.json').read_text())
    assert all(sha(p)==v for p,v in hashes.items())
    lines=['# PCA plus ridge on mapped genes','',
        '528 participants, fixed 4,376 mapped candidates and the same 30 nested outer folds. Training-only PCA retains 10,25,50 or 100 components, jointly tuned with the original C grid (.001,.01,.1,1). Unwhitened PC scores preserve their training variances.', '',
        '| Protocol | Model | Held-out AUROC (repeat range) | Training AUROC | Train–test gap |',
        '| --- | --- | --- | ---: | ---: |']
    for r in summary.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.variant} | {r.AUROC_mean:.3f} ({r.AUROC_min:.3f}–{r.AUROC_max:.3f}) | {r.train_AUROC_mean:.3f} | {r.train_test_gap_mean:.3f} |')
    lines += ['', 'Primary AUROC averages within-fold AUROCs, then three repeat means. Repeat ranges are descriptive, not confidence intervals. Paired PCA-minus-original differences are in paired_summary.tsv. Secondary probability and threshold metrics are separately labeled.', '',
        'Original ridge inner scores, selected C/threshold and outer predictions reproduced in all 30 folds. Cohort/count integrity, matched participants/genes, nested boundaries, CPU/RTX preprocessing, PCA orthonormality, held-out perturbation, label-independent PCA, convergence, collapsed gene-coefficient inference, serialization and source hashes passed.', '',
        'A full-data PCA/ridge development artifact is saved, including gene preprocessing, PCA loadings and classifier. Equivalent standardized-gene coefficients are algebraic representations of PCA/ridge and are not gene-selection frequencies.', '',
        'This is an adaptive exploratory comparison after observing previous model results. PCA preserves high variance, which need not be disease signal. The current ridge remains the benchmark; independent-cohort validation is outstanding.', '',
        '![Comparison](pca_ridge_comparison.png)','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    save('summary.json',dict(status='COMPLETE',participants=528,candidate_genes=4376,final_eligible_genes=eligible_count,
        primary_results=summary.to_dict(orient='records'),validation='PASS',benchmark_reproduction_folds=30,
        source_hashes_unchanged=True,external_validation=False))
    status('COMPLETE',report='RESULTS.md')


if __name__=='__main__':
    if '--run' not in sys.argv and (ROOT/'status.json').exists(): raise SystemExit('Existing run; refusing duplicate launch')
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            if '--run' in sys.argv: train()
            else:
                checked=prepare(); env=os.environ.copy()
                env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
                (ROOT/'matplotlib_cache').mkdir(exist_ok=True); status('LAUNCHING')
                with (ROOT/'run.log').open('ab',buffering=0) as log:
                    job=subprocess.Popen([sys.executable,'-u',str(ROOT/'launch.py'),'--run'],stdin=subprocess.DEVNULL,stdout=log,
                                         stderr=subprocess.STDOUT,cwd=ROOT.parent.parent,env=env,start_new_session=True)
                save('job.json',dict(pid=job.pid,device=checked['device_name'],log=str(ROOT/'run.log')))
                print(json.dumps(dict(launched_pid=job.pid,preflight=checked),indent=2))
    except Exception:
        status('FAILED',error=traceback.format_exc()); raise
