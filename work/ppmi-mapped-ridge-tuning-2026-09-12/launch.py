"""Fine regularization tuning with a reproduced coarse ridge control."""
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
from ridge_tuning import (BASE, PRIOR, HALLMARK, MAPPED, MappedData, choose_device,
    make_splits, SEED, PENALTIES, COARSE_CS, FINE_CS, fit_model, tune, artifact, predict_artifact, evaluate, validate)

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
    paths += [MAPPED / n for n in ['selected_parameters.tsv', 'inner_tuning_scores.tsv']]
    paths += [ROOT / n for n in ['README.md', 'ridge_tuning.py', 'launch.py', 'predict.py', 'candidate_genes.tsv', 'fold_plan.json']]
    save('input_sha256.json', {str(p): sha(p) for p in paths})
    return checked



def summarize(pred, fold_metrics):
    fine = pd.DataFrame(fold_metrics).assign(variant='fine_grid')
    old = pd.read_csv(MAPPED/'outer_fold_metrics.tsv', sep='\t')
    old = old[old.penalty=='ridge'].assign(variant='original_grid')
    paired = fine.merge(old, on=KEYS, suffixes=('_fine', '_original'), validate='one_to_one')
    assert len(paired)==30
    for name in ['n_train','n_test','eligible_genes']:
        assert np.array_equal(paired[name+'_fine'],paired[name+'_original'])
    paired['AUROC_difference'] = paired.test_AUROC_fine-paired.test_AUROC_original
    table('paired_fold_comparisons.tsv',paired)
    per_repeat = paired.groupby(['repeat','protocol']).AUROC_difference.mean().reset_index()
    table('paired_repeat_comparisons.tsv',per_repeat)
    table('paired_summary.tsv',per_repeat.groupby('protocol').AUROC_difference.agg(['mean','min','max']).reset_index())
    combined = pd.concat([fine,old],ignore_index=True)
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


def finalize(summary, eligible_count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    repeats=pd.read_csv(ROOT/'primary_per_repeat.tsv',sep='\t')
    params=pd.read_csv(ROOT/'selected_parameters.tsv',sep='\t')
    table('selected_C_frequency.tsv',params.groupby(['protocol','C']).size().rename('folds').reset_index())
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,protocol in zip(axes,['participant_stratified','batch_grouped']):
        for repeat in sorted(repeats.repeat.unique()):
            values=repeats[(repeats.protocol==protocol)&(repeats.repeat==repeat)].set_index('variant').loc[['original_grid','fine_grid'],'AUROC']
            ax.plot([0,1],values,marker='o',label=f'Repeat {repeat}')
        ax.axhline(.5,color='gray',ls='--')
        ax.set(xticks=[0,1],xticklabels=['Original grid','Fine grid'],ylim=(0,1),title=protocol)
    axes[0].set_ylabel('Mean within-fold AUROC per repeat')
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT/'ridge_tuning_comparison.png',dpi=170); plt.close(fig)
    hashes=json.loads((ROOT/'input_sha256.json').read_text())
    assert all(sha(p)==v for p,v in hashes.items())
    lines=['# Finer mapped-gene ridge tuning','',
        'Same 528 participants, fixed 4,376 mapped candidates and all training-eligible individual genes. The only predictive change is expanding/filling the C grid from four to eleven values, with nested inner-fold selection.', '',
        '| Protocol | Grid | Held-out AUROC (repeat range) | Training AUROC | Train–test gap |',
        '| --- | --- | --- | ---: | ---: |']
    for r in summary.itertuples(index=False):
        lines.append(f'| {r.protocol} | {r.variant} | {r.AUROC_mean:.3f} ({r.AUROC_min:.3f}–{r.AUROC_max:.3f}) | {r.train_AUROC_mean:.3f} | {r.train_test_gap_mean:.3f} |')
    lines += ['', 'Primary AUROC is the mean within-fold AUROC, averaged over three repeats. Ranges are descriptive, not confidence intervals. Fine-minus-original paired results are in paired_summary.tsv. Probability/threshold metrics are secondary.', '',
        'All 30 original coarse-grid inner score sets, selected C values, thresholds and outer prediction vectors reproduced within recorded numerical tolerances. Exact participant/label and gene identities matched; count/cohort, nested boundaries, CPU/RTX agreement, held-out perturbation, convergence, inference serialization and source-hash checks passed.', '',
        'One full-data fine-grid ridge development artifact is saved. Any gain is exploratory: the grid was expanded after examining previous analyses on this cohort, and independent validation remains outstanding. More tuning can raise inner scores without improving held-out discrimination.', '',
        '![Paired comparison](ridge_tuning_comparison.png)','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    save('summary.json',dict(status='COMPLETE',participants=528,candidate_genes=4376,final_eligible_genes=eligible_count,
        primary_results=summary.to_dict(orient='records'),validation='PASS',benchmark_reproduction_folds=30,
        source_hashes_unchanged=True,external_validation=False))
    status('COMPLETE',report='RESULTS.md')


def train():
    raw,genes=np.load(BASE/'counts.npy'),np.load(BASE/'gene_ids.npy')
    indices=np.load(HALLMARK/'membership.npz')['indices']
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str}); y=meta.label.to_numpy()
    data=MappedData(raw,genes,y,indices,choose_device())
    plan=json.loads((ROOT/'fold_plan.json').read_text())
    old_params = pd.read_csv(MAPPED/'selected_parameters.tsv', sep='\t')
    old_params = old_params[old_params.penalty=='ridge'].set_index(KEYS)
    old_inner = pd.read_csv(MAPPED/'inner_tuning_scores.tsv', sep='\t')
    old_inner = old_inner[old_inner.penalty=='ridge'].set_index(KEYS+['C'])
    old_pred = pd.read_csv(MAPPED/'out_of_fold_predictions.tsv', sep='\t', dtype={'PATNO':str})
    old_pred = old_pred[old_pred.penalty=='ridge']
    old_coefs = pd.read_csv(MAPPED/'outer_gene_coefficients.tsv', sep='\t', usecols=KEYS+['penalty','Geneid'])
    feature_sets = {k:g.Geneid.to_numpy() for k,g in old_coefs[old_coefs.penalty=='ridge'].groupby(KEYS)}
    test_sets = {k:g.set_index('PATNO') for k,g in old_pred.groupby(KEYS)}
    predictions,metrics,params,tuning,coefficients=[],[],[],[],[]
    checks=[]
    for number,split in enumerate(plan,1):
        context={k:split[k] for k in ['repeat','protocol','fold']}
        status('TRAINING',completed_folds=number-1,total_folds=30,**context)
        print(f'{datetime.now(timezone.utc).isoformat()} {number}/30 {context}',flush=True)
        tr,te=np.array(split['train_indices']),np.array(split['test_indices'])
        inner=[(np.array(s['train_positions']),np.array(s['validation_positions'])) for s in split['inner']]
        best,rows=tune(data,tr,inner); tuning.extend(dict(**context,**r) for r in rows)
        pack=data.pack(tr,te)
        assert set(pack['state']['indices'])<=set(indices)
        key=tuple(context[k] for k in KEYS)
        assert np.array_equal(genes[pack['state']['indices']], feature_sets[key])
        previous=test_sets[key].loc[meta.PATNO.iloc[te]]
        assert len(previous)==len(te) and np.array_equal(previous.label,y[te])
        chosen=best['ridge']; old_chosen=old_params.loc[key]
        assert chosen['coarse_C']==old_chosen.C
        np.testing.assert_allclose(chosen['coarse_threshold'],old_chosen.threshold,atol=1e-10,rtol=1e-10)
        for row in rows:
            if row['C'] in COARSE_CS:
                reference=old_inner.loc[key+(row['C'],)]
                np.testing.assert_allclose(row['mean_inner_AUROC'],reference.mean_inner_AUROC,atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.fromstring(row['inner_AUROCs'],sep=';'),np.fromstring(reference.inner_AUROCs,sep=';'),atol=1e-12,rtol=0)
        control=fit_model(pack['train'],y[tr],chosen['coarse_C'],'ridge',0.)
        control_p=control.predict_proba(pack['test'])[:,1]
        np.testing.assert_allclose(control_p,previous.probability_PD.to_numpy(),atol=1e-8,rtol=1e-8)
        checks.append(dict(**context,coarse_C=chosen['coarse_C'],fine_C=chosen['C'],
                          max_probability_difference=float(np.max(np.abs(control_p-previous.probability_PD.to_numpy()))),status='PASS'))
        table('benchmark_reproduction.tsv',checks)
        for penalty,chosen in best.items():
            model=fit_model(pack['train'],y[tr],chosen['C'],penalty,chosen['l1_ratio'])
            p=model.predict_proba(pack['test'])[:,1]
            predictions.extend(dict(**context,penalty=penalty,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(prob),
                                    threshold=chosen['threshold']) for i,prob in zip(te,p))
            metrics.append(dict(**context,penalty=penalty,n_train=len(tr),n_test=len(te),eligible_genes=pack['train'].shape[1],
                                nonzero_genes=int(np.count_nonzero(model.coef_)),
                                train_AUROC=float(roc_auc_score(y[tr],model.predict_proba(pack['train'])[:,1])),test_AUROC=float(roc_auc_score(y[te],p))))
            params.append(dict(**context,**chosen,eligible_genes=pack['train'].shape[1],
                               optimizer_iterations=int(getattr(model,'total_optimizer_iterations_',model.n_iter_[0]))))
            state=artifact(model,pack,chosen,genes)
            np.testing.assert_allclose(predict_artifact(state,raw[te[:3]]),p[:3],atol=1e-9,rtol=1e-9)
            coefficients.extend(dict(**context,penalty=penalty,Geneid=g,coefficient=float(c),nonzero=bool(c!=0)) for g,c in zip(state['selected_gene_ids'],model.coef_[0]))
        table('out_of_fold_predictions.tsv',predictions); table('selected_parameters.tsv',params); table('inner_tuning_scores.tsv',tuning); table('outer_fold_metrics.tsv',metrics)
    pred=pd.DataFrame(predictions)
    assert len(pred)==528*3*2 and not pred.duplicated(['repeat','protocol','penalty','PATNO']).any()
    status('SUMMARIZING')
    summary=summarize(pred,metrics)
    coef=pd.DataFrame(coefficients); table('outer_gene_coefficients.tsv',coef)
    stable=coef.groupby(['protocol','penalty','Geneid']).agg(eligible_fits=('coefficient','size'),nonzero_fits=('nonzero','sum'),
                  positive_fits=('coefficient',lambda x:int((x>0).sum())),negative_fits=('coefficient',lambda x:int((x<0).sum())),
                  mean_coefficient_when_eligible=('coefficient','mean')).reset_index()
    stable['total_fits']=15; assert (stable.eligible_fits<=15).all()
    table('gene_retention.tsv',stable)
    status('FITTING_FINAL_DEVELOPMENT_MODELS')
    splits=make_splits(y,meta.batch.to_numpy(str),'participant_stratified',5,SEED+9901)
    save('final_tuning_folds.json',[dict(train_indices=a.tolist(),validation_indices=b.tolist()) for a,b in splits])
    best,rows=tune(data,np.arange(528),splits); table('final_tuning_scores.tsv',rows)
    full=data.pack(np.arange(528),np.array([],dtype=int)); (ROOT/'models').mkdir(exist_ok=True)
    for penalty,chosen in best.items():
        model=fit_model(full['train'],y,chosen['C'],penalty,chosen['l1_ratio'])
        state=artifact(model,full,chosen,genes)
        state.update(training_n=528,candidate_gene_ids=genes[indices],sklearn_version=sklearn.__version__,numpy_version=np.__version__)
        path=ROOT/'models'/f'mapped_RNA_{penalty}.joblib'; joblib.dump(state,path)
        np.testing.assert_allclose(predict_artifact(joblib.load(path),raw[:5]),model.predict_proba(full['train'][:5])[:,1],atol=1e-9)
    save('final_parameters.json',list(best.values()))
    finalize(summary, full['train'].shape[1])


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
