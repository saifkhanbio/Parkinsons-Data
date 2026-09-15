"""Descriptive expression, covariate, and influence audit of two selected genes."""
from pathlib import Path
import hashlib
import itertools
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BASE = WORK / 'ppmi-classifier-2026-09-12'
REFINE = WORK / 'ppmi-classifier-refinement-retry-2026-09-12'
sys.path.insert(0, str(REFINE))
from refine import fit_model, predict_artifact
from classifier import clinical_matrix

GENES = {'ENSG00000161040.16': 'FBXL13', 'ENSG00000230257.2': 'NFE4'}
CELL = ['log_wbc', 'neutrophils_percent', 'monocytes_percent', 'eosinophils_percent', 'basophils_percent']
TECH = ['RIN', 'intergenic_percent', 'log_library']
MODES = ['unadjusted', 'demographic', 'technical', 'cell_adjusted']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(name, data):
    data.to_csv(ROOT / name, sep='\t', index=False)


def design(meta, mode):
    x = pd.DataFrame({'intercept': np.ones(len(meta)), 'PD': meta.label.to_numpy(float)})
    blocks = {'PD': ['PD']}
    if mode != 'unadjusted':
        x['age'] = meta.age_collection_years.to_numpy(float)
        x['male'] = meta.sex.eq('Male').to_numpy(float)
        blocks['demographic'] = ['age', 'male']
    if mode in ['technical', 'cell_adjusted']:
        batch = pd.get_dummies(meta.batch, prefix='batch', drop_first=True, dtype=float).reset_index(drop=True)
        x = pd.concat([x, batch], axis=1)
        for col in TECH:
            x[col] = meta[col].to_numpy(float)
        blocks['batch'], blocks['RNA_quality_and_library'] = batch.columns.tolist(), TECH
    if mode == 'cell_adjusted':
        for col in CELL:
            x[col] = meta[col].to_numpy(float)
        blocks['blood_cells'] = CELL
    for col in ['age'] + TECH + CELL:
        if col in x:
            scale = x[col].std(ddof=0)
            assert scale > 0
            x[col] = (x[col]-x[col].mean())/scale
    assert np.isfinite(x.to_numpy()).all()
    assert np.linalg.matrix_rank(x) == x.shape[1]
    return x, blocks


def main():
    assert not (ROOT / 'summary.json').exists(), 'Existing audit must be preserved.'
    paths = [BASE/'counts.npy', BASE/'gene_ids.npy', BASE/'metadata.tsv', BASE/'input_sha256.json',
             WORK/'ppmi-gene-annotation-review-2026-09-12/gencode_v29_gene_reference.tsv',
             WORK/'ppmi-classifier-feature-stability-2026-09-12/elasticnet_core.tsv',
             REFINE/'refine.py', REFINE/'elastic_solver.py', BASE/'classifier.py', ROOT/'README.md', ROOT/'audit.py']
    paths += [REFINE/'models'/f'{f}_{p}.joblib' for f,p in itertools.product(['rna','combined'], ['ridge','elasticnet'])]
    de_paths = [(n, model, WORK/d/model/'results.tsv') for n,d in [(528,'ppmi-blood-cell-adjustment-2026-09-12'),
              (445,'ppmi-blood-cell-timing-2026-09-12')] for model in ['subset_reference','cell_adjusted']]
    paths += [p for _,_,p in de_paths]
    hashes = {str(p): sha(p) for p in paths}
    original = json.loads((BASE/'input_sha256.json').read_text())
    for p in paths[:3]:
        assert hashes[str(p)] == original[str(p)]
    meta = pd.read_csv(BASE/'metadata.tsv', sep='\t', dtype={'PATNO': str})
    genes, raw = np.load(BASE/'gene_ids.npy'), np.load(BASE/'counts.npy')
    assert meta.PATNO.is_unique and len(meta)==528 and raw.shape==(528,58780)
    assert np.issubdtype(raw.dtype, np.integer) and (raw>=0).all()
    assert meta.group.value_counts().to_dict()=={'PD':358,'Control':170}
    assert np.array_equal(meta.label, meta.group.eq('PD').astype(int))
    library = raw.sum(1, dtype=np.int64)
    assert (library>0).all() and (meta.wbc>0).all()
    meta['log_library'], meta['log_wbc'] = np.log(library), np.log(meta.wbc)
    meta['male'], meta['phase2'] = meta.sex.eq('Male').astype(int), meta.phase.eq('PPMI-Phase2').astype(int)
    annotation = pd.read_csv(paths[4], sep='\t')
    annotation = annotation[annotation.reference_Geneid.isin(GENES)]
    assert len(annotation)==2 and annotation.reference_Geneid.is_unique
    assert annotation.set_index('reference_Geneid').reference_gene_name.to_dict()==GENES
    save('annotations.tsv', annotation)
    core = pd.read_csv(paths[5], sep='\t')
    assert set(core.Geneid)==set(GENES)
    batch_sizes = meta.groupby('batch').agg(n=('PATNO','size'), PD=('label','sum'), phase=('phase','first')).reset_index()
    batch_sizes['Control'] = batch_sizes.n-batch_sizes.PD
    save('batch_sizes.tsv', batch_sizes)
    de = []
    for n, model, path in de_paths:
        selected = pd.read_csv(path, sep='\t').query('Geneid in @GENES').copy()
        assert len(selected)==2
        selected['participants'], selected['model'] = n, model
        selected['gene'] = selected.Geneid.map(GENES)
        selected['fold_change_PD_vs_control'] = np.exp2(selected.log2FoldChange)
        selected['fold_change_CI_low'] = np.exp2(selected.log2FoldChange-1.96*selected.lfcSE)
        selected['fold_change_CI_high'] = np.exp2(selected.log2FoldChange+1.96*selected.lfcSE)
        de.append(selected)
    de = pd.concat(de, ignore_index=True)
    save('existing_deseq2_evidence.tsv', de)
    distribution, correlation, effects, blocks_out, influence, omissions, batch_omissions, sample_rows = [],[],[],[],[],[],[],[]
    targets, expression = {}, {}
    covariates = ['age_collection_years','male','wbc','neutrophils_percent','lymphocytes_percent',
                  'monocytes_percent','eosinophils_percent','basophils_percent',*TECH,'phase2']
    for gene, name in GENES.items():
        index = np.flatnonzero(genes==gene)
        assert len(index)==1
        counts = raw[:,index[0]]
        log_cpm = np.log2(1+counts/library*1e6)
        expression[name] = log_cpm
        for group in ['All','Control','PD']:
            use = np.ones(528,dtype=bool) if group=='All' else meta.group.eq(group).to_numpy()
            c, z = counts[use], log_cpm[use]
            distribution.append(dict(Geneid=gene,gene=name,group=group,n=int(use.sum()),detected=int((c>0).sum()),
                count_ge_10=int((c>=10).sum()),raw_min=int(c.min()),raw_q25=float(np.quantile(c,.25)),raw_median=float(np.median(c)),
                raw_q75=float(np.quantile(c,.75)),raw_max=int(c.max()),log_CPM_mean=float(z.mean()),log_CPM_SD=float(z.std(ddof=1)),
                log_CPM_median=float(np.median(z)),log_CPM_min=float(z.min()),log_CPM_max=float(z.max())))
            for col in covariates:
                values = meta.loc[use,col].to_numpy(float)
                assert np.isfinite(values).all()
                r = float(spearmanr(z,values).statistic) if np.unique(values).size>1 else np.nan
                correlation.append(dict(Geneid=gene,gene=name,group=group,covariate=col,method='Spearman',r=r))
        # Partial Spearman for blood covariates: rank continuous quantities first.
        nuisance = pd.DataFrame(dict(intercept=np.ones(528),PD=meta.label,male=meta.male))
        for col in ['age_collection_years',*TECH]:
            nuisance[col] = rankdata(meta[col])
        nuisance = pd.concat([nuisance,pd.get_dummies(meta.batch,drop_first=True,dtype=float)],axis=1)
        assert np.linalg.matrix_rank(nuisance)==nuisance.shape[1]
        q,_ = np.linalg.qr(nuisance.to_numpy())
        residual_y = rankdata(log_cpm)-q@(q.T@rankdata(log_cpm))
        for col in ['wbc','neutrophils_percent','lymphocytes_percent','monocytes_percent','eosinophils_percent','basophils_percent']:
            ranked = rankdata(meta[col]); residual_x = ranked-q@(q.T@ranked)
            correlation.append(dict(Geneid=gene,gene=name,group='All',covariate=col,
                                    method='Partial Spearman: diagnosis+demographic+technical+batch',
                                    r=float(np.corrcoef(residual_y,residual_x)[0,1])))
        for mode in MODES:
            x, blocks = design(meta,mode)
            fit = sm.OLS(log_cpm,x).fit()
            inv = fit.normalized_cov_params.to_numpy()
            leverage = np.sum(x.to_numpy()*(x.to_numpy()@inv),axis=1)
            unit = leverage>=1-1e-8
            interval = fit.get_robustcov_results(cov_type='HC3').conf_int()[x.columns.get_loc('PD')] if not unit.any() else [np.nan,np.nan]
            effects.append(dict(Geneid=gene,gene=name,model=mode,n=528,parameters=x.shape[1],rank=int(np.linalg.matrix_rank(x)),
                                condition_number=float(np.linalg.cond(x)),PD_coefficient=float(fit.params.PD),
                                HC3_CI_low=float(interval[0]),HC3_CI_high=float(interval[1]),
                                HC3_available=not bool(unit.any()),unit_leverage_rows=int(unit.sum()),adjusted_R2=float(fit.rsquared_adj)))
            if mode!='cell_adjusted':
                continue
            for block, columns in blocks.items():
                reduced = sm.OLS(log_cpm,x.drop(columns=columns)).fit()
                blocks_out.append(dict(Geneid=gene,gene=name,block=block,partial_R2=float((reduced.ssr-fit.ssr)/reduced.ssr)))
            residual = np.asarray(fit.resid)
            cook = np.full(528,np.nan); student = np.full(528,np.nan)
            regular = ~unit
            student[regular] = residual[regular]/np.sqrt(fit.mse_resid*(1-leverage[regular]))
            cook[regular] = student[regular]**2*leverage[regular]/(x.shape[1]*(1-leverage[regular]))
            group_col = x.columns.get_loc('PD')
            beta = float(fit.params.PD)
            deleted = np.full(528,np.nan)
            deleted[regular] = beta-((x.to_numpy()@inv[:,group_col])[regular]*residual[regular]/(1-leverage[regular]))
            # Direct refits cover non-estimable fixed-design deletions and check the formula.
            checks = set(np.flatnonzero(unit)) | {int(np.nanargmax(cook)),int(np.argmax(log_cpm)),int(np.nanargmax(np.abs(deleted-beta)))}
            for i in sorted(checks):
                keep = np.arange(528)!=i
                xx,_ = design(meta.loc[keep].reset_index(drop=True),'cell_adjusted')
                direct = float(sm.OLS(log_cpm[keep],xx).fit().params.PD)
                if regular[i]:
                    np.testing.assert_allclose(deleted[i],direct,atol=1e-8,rtol=1e-8)
                else:
                    deleted[i] = direct
            assert np.isfinite(deleted).all()
            for i in range(528):
                source = dict(Geneid=gene,gene=name,PATNO=meta.PATNO.iloc[i],group=meta.group.iloc[i],batch=meta.batch.iloc[i])
                influence.append(dict(**source,leverage=float(leverage[i]),studentized_residual=float(student[i]),
                                      cook_distance=float(cook[i]),unit_leverage=bool(unit[i])))
                omissions.append(dict(**source,full_PD_coefficient=beta,deleted_PD_coefficient=float(deleted[i]),
                                      change=float(deleted[i]-beta),direction_changed=bool(np.sign(deleted[i])!=np.sign(beta))))
            targets.setdefault(int(np.nanargmax(cook)),[]).append(name+':maximum_OLS_Cook')
            targets.setdefault(int(np.argmax(log_cpm)),[]).append(name+':maximum_logCPM')
            for batch in sorted(meta.batch.unique()):
                keep = meta.batch.ne(batch).to_numpy()
                xx,_ = design(meta.loc[keep].reset_index(drop=True),'cell_adjusted')
                direct = float(sm.OLS(log_cpm[keep],xx).fit().params.PD)
                batch_omissions.append(dict(Geneid=gene,gene=name,omitted_batch=batch,n_removed=int((~keep).sum()),
                                            full_PD_coefficient=beta,deleted_PD_coefficient=direct,change=direct-beta,
                                            direction_changed=bool(np.sign(direct)!=np.sign(beta))))
        samples = meta.copy()
        samples['Geneid'],samples['gene'],samples['raw_count'],samples['log2_1_plus_CPM'] = gene,name,counts,log_cpm
        sample_rows.append(samples)
    distribution,correlation,effects = pd.DataFrame(distribution),pd.DataFrame(correlation),pd.DataFrame(effects)
    influence,omissions,batch_omissions = pd.DataFrame(influence),pd.DataFrame(omissions),pd.DataFrame(batch_omissions)
    save('expression_distributions.tsv',distribution); save('covariate_correlations.tsv',correlation)
    save('OLS_effects.tsv',effects); save('conditional_partial_R2.tsv',pd.DataFrame(blocks_out))
    save('sample_influence.tsv',influence); save('all_participant_omissions.tsv',omissions)
    save('all_batch_omissions.tsv',batch_omissions); save('sample_expression_and_covariates.tsv',pd.concat(sample_rows))
    save('targeted_classifier_omissions.tsv',pd.DataFrame([dict(PATNO=meta.PATNO.iloc[i],reason=';'.join(reason)) for i,reason in targets.items()]))
    # Conditional coefficient checks of final development artifacts: no threshold or performance evaluation.
    classifier_rows = []
    for family,penalty in itertools.product(['rna','combined'],['ridge','elasticnet']):
        state = joblib.load(REFINE/'models'/f'{family}_{penalty}.joblib')
        assert np.array_equal(state['feature_universe'],genes)
        pieces = []
        if 'clinical' in state:
            s = state['clinical']
            values = clinical_matrix(meta,state['clinical_family'],library)
            pieces.append((values-s['means'])/s['scales'])
        s = state['rna']
        values = np.log2(1+raw[:,s['indices']]/library[:,None]*1e6)
        pieces.append((values-s['means'])/s['scales'])
        x = np.column_stack(pieces)
        model = state['classifier']
        np.testing.assert_allclose(model.predict_proba(x[:5])[:,1],predict_artifact(state,raw[:5],meta.iloc[:5]),atol=1e-9)
        offset = len(state['clinical']['features']) if 'clinical' in state else 0
        positions = {g: np.flatnonzero(state['selected_gene_ids']==g) for g in GENES}
        for i,reason in targets.items():
            keep = np.arange(528)!=i
            chosen = state['hyperparameters']
            refit = fit_model(x[keep],meta.label.to_numpy()[keep],chosen['C'],penalty,chosen['l1_ratio'])
            for gene,name in GENES.items():
                selected = len(positions[gene])==1
                j = int(positions[gene][0])+offset if selected else None
                before = float(model.coef_[0,j]) if selected else 0.
                after = float(refit.coef_[0,j]) if selected else 0.
                classifier_rows.append(dict(gene=name,Geneid=gene,family=family,penalty=penalty,
                      omitted_PATNO=meta.PATNO.iloc[i],reason=';'.join(reason),selected_in_final_model=selected,
                      full_coefficient=before,deleted_coefficient=after,change=after-before,
                      relative_absolute_change=abs(after-before)/abs(before) if before else np.nan,
                      direction_changed=bool(np.sign(after)!=np.sign(before)),retained_after=after!=0))
    classifier = pd.DataFrame(classifier_rows)
    save('classifier_conditional_omission_checks.tsv',classifier)
    # Figures use aggregate displays and deidentified scatter points.
    os.environ['MPLCONFIGDIR'] = str(ROOT/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20260912)
    fig,axes = plt.subplots(2,3,figsize=(13,8))
    for row,(gene,name) in enumerate(GENES.items()):
        z = expression[name]
        for j,group in enumerate(['Control','PD']):
            values = z[meta.group.eq(group)]
            axes[row,0].scatter(j+rng.normal(0,.055,len(values)),values,s=8,alpha=.4)
        axes[row,0].set(xticks=[0,1],xticklabels=['Control','PD'],ylabel='log2(1+CPM)',title=name)
        overall = correlation[(correlation.gene==name)&(correlation.group=='All')&(correlation.method=='Spearman')]
        cell = overall[overall.covariate.isin(['wbc','neutrophils_percent','lymphocytes_percent','monocytes_percent','eosinophils_percent','basophils_percent'])]
        strongest = cell.iloc[cell.r.abs().argmax()]
        for group in ['Control','PD']:
            use = meta.group.eq(group)
            axes[row,1].scatter(meta.loc[use,strongest.covariate],z[use],s=9,alpha=.5,label=group)
        axes[row,1].set(xlabel=strongest.covariate,ylabel='log2(1+CPM)',title=f'{name}: Spearman r={strongest.r:.2f}')
        axes[row,1].legend(fontsize=8)
        a = de[(de.Geneid==gene)&(de.participants==528)]
        for j,mode in enumerate(['subset_reference','cell_adjusted']):
            r = a[a.model==mode].iloc[0]
            axes[row,2].errorbar(j,r.log2FoldChange,yerr=1.96*r.lfcSE,fmt='o',capsize=4)
        axes[row,2].axhline(0,color='gray',ls='--')
        axes[row,2].set(xticks=[0,1],xticklabels=['Reference','CBC-adjusted'],ylabel='DESeq2 log2 fold change (95% Wald CI)',title=name+' — existing 528-person results')
    fig.tight_layout(); fig.savefig(ROOT/'expression_and_confounding.png',dpi=170); plt.close(fig)
    assert all(sha(Path(p))==v for p,v in hashes.items()),'Source changed'
    summary = dict(status='COMPLETE',participants=528,genes=GENES,validation='PASS',source_hashes_unchanged=True,
                   individual_omission_checks=len(omissions),batch_omission_checks=len(batch_omissions),
                   targeted_classifier_refits=len(targets)*4,classifier_gene_coefficient_checks=len(classifier),
                   OLS_individual_direction_changes=int(omissions.direction_changed.sum()),
                   OLS_batch_direction_changes=int(batch_omissions.direction_changed.sum()),
                   classifier_direction_changes=int(classifier.direction_changed.sum()),
                   independent_validation=False)
    lines = ['# FBXL13 and NFE4 expression and confounding audit','',
             'Descriptive audit of the two stability-selected genes in the existing 528-person cohort. No new classifier performance estimate or external validation is produced.','',
             '| Gene | Detected | Raw-count median (range) | Reference PD/control FC | CBC-adjusted FC | CBC-adjusted genome-wide FDR |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    for gene,name in GENES.items():
        d = distribution[(distribution.Geneid==gene)&(distribution.group=='All')].iloc[0]
        a = de[(de.Geneid==gene)&(de.participants==528)].set_index('model')
        lines.append(f'| {name} | {d.detected}/528 | {d.raw_median:g} ({d.raw_min}–{d.raw_max}) | {a.loc["subset_reference","fold_change_PD_vs_control"]:.3f} | {a.loc["cell_adjusted","fold_change_PD_vs_control"]:.3f} | {a.loc["cell_adjusted","padj"]:.3f} |')
    lines += ['', 'Both genes are reliably detected in this dataset. Blood-cell adjustment reduces their estimated disease association; both retain positive effects but neither meets the original genome-wide FDR<0.05 criterion after adjustment. The smaller 445-person timing analysis is included in existing_deseq2_evidence.tsv. Loss of significance alone does not establish absence of association or a causal cell-composition explanation.', '',
              '## Covariate associations and influence', '']
    for gene,name in GENES.items():
        cell = correlation[(correlation.Geneid==gene)&(correlation.group=='All')&(correlation.method=='Spearman')&correlation.covariate.isin(['wbc','neutrophils_percent','lymphocytes_percent','monocytes_percent','eosinophils_percent','basophils_percent'])]
        strongest = cell.iloc[cell.r.abs().argmax()]
        partial = correlation[(correlation.Geneid==gene)&correlation.method.str.startswith('Partial')]
        strongest_partial = partial.iloc[partial.r.abs().argmax()]
        omit = omissions[omissions.Geneid==gene]
        batch = batch_omissions[batch_omissions.Geneid==gene]
        checks = classifier[classifier.Geneid==gene]
        lines.append(f'- **{name}:** strongest unadjusted blood-cell correlation was {strongest.covariate} (Spearman r={strongest.r:.3f}); strongest partial blood-cell correlation was {strongest_partial.covariate} (r={strongest_partial.r:.3f}). Fully adjusted OLS PD coefficient {omit.full_PD_coefficient.iloc[0]:.3f}; individual-deletion range {omit.deleted_PD_coefficient.min():.3f}–{omit.deleted_PD_coefficient.max():.3f}; whole-batch-deletion range {batch.deleted_PD_coefficient.min():.3f}–{batch.deleted_PD_coefficient.max():.3f}. Across targeted final-classifier omissions, maximum relative coefficient change was {checks.relative_absolute_change.max():.1%}; sign/nonzero changes: {int(checks.direction_changed.sum())}.')
    lines += ['', 'OLS coefficients above refer to log2(1+CPM), not DESeq2 log2 fold changes. Partial rank correlations control diagnosis, age, sex, RNA quality, library size, and batch. Covariate-block partial R-squared, sex/phase correlations, and within-diagnosis correlations are saved separately.', '',
              'Individual OLS deletions cover every participant. Targeted classifier deletions cover only the maximum logCPM and maximum estimable OLS Cook observations for each gene; they preserve feature selection, scaling and tuning. They cannot rule out broader pipeline influence. Singleton batch levels may create unit leverage; corresponding Cook/residual quantities and HC3 intervals are marked unavailable, and deletions are directly refitted after dropping unused batch levels.', '',
              'These results assess robustness and confounding patterns in the development data. They do not establish cell-independent effects, disease specificity, causal biology, or a validated two-gene panel. Month-level screening CBC measurements may differ from cell composition at RNA collection. Coefficient recurrence and influence robustness do not overcome modest predictive performance.', '',
              'All input, annotation, design-rank, deletion-formula, model-inference, optimizer and source-integrity checks passed. Restricted per-participant evidence stays in local TSVs.','',
              '![Expression and covariates](expression_and_confounding.png)','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    (ROOT/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
