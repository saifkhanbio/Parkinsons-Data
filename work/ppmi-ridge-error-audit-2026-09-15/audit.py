"""Audit saved predictions only; no RNA fitting or source mutation."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
import json,hashlib,time,traceback
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit,logit
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
BASE=ROOT.parent/'ppmi-classifier-2026-09-12'
MAPPED=ROOT.parent/'ppmi-mapped-gene-classifier-2026-09-12'
BENCH=ROOT.parent/'ppmi-hallmark-blood-block-ridge-2026-09-15'
PROTOCOLS=['participant_stratified','batch_grouped']
BOOT=2000

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def table(name,rows):pd.DataFrame(rows).to_csv(ROOT/name,sep='\t',index=False)

def status(stage,**kwargs):
    save('status.json',dict(status=stage,**kwargs));print(stage,flush=True)

def weights(meta,protocol):
    rng=np.random.default_rng(20260915+PROTOCOLS.index(protocol))
    if protocol=='participant_stratified':
        w=np.zeros((BOOT,len(meta)))
        for y in [0,1]:
            ids=np.flatnonzero(meta.label.to_numpy()==y)
            w[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=BOOT)
    else:
        lv,code=np.unique(meta.batch,return_inverse=True)
        w=rng.multinomial(len(lv),np.full(len(lv),1/len(lv)),size=BOOT)[:,code].astype(float)
    return np.vstack([np.ones(len(meta)),w])

def calibration(y,p,w,offset=False):
    """Batched weighted Newton logistic regressions, two or one coefficients."""
    z=logit(np.clip(p,1e-12,1-1e-12));result=[]
    for first in range(0,len(w),200):
        ww=w[first:first+200];n=len(ww)
        a=np.zeros(n);b=np.ones(n) if offset else np.zeros(n)
        for iteration in range(100):
            mu=expit(a[:,None]+b[:,None]*z)
            v=ww*np.maximum(mu*(1-mu),1e-14);e=ww*(y-mu)
            h0=v.sum(1);g0=e.sum(1)
            if offset:
                da=g0/h0;db=np.zeros(n)
            else:
                h1=v@z;h2=v@(z*z);g1=e@z;det=h0*h2-h1*h1
                assert np.all(det>1e-10),'Calibration singularity'
                da=(g0*h2-g1*h1)/det;db=(g1*h0-g0*h1)/det
            delta=max(np.max(abs(da)),np.max(abs(db)))
            scale=np.maximum(1,np.maximum(abs(da),abs(db))/2)
            a+=da/scale;b+=db/scale
            if delta<1e-9:break
        else:raise AssertionError('Calibration did not converge')
        result.append(np.c_[a,b])
    return np.concatenate(result)

def interval(protocol,metric,v):
    assert np.isfinite(v).all()
    lo,hi=np.quantile(v[1:],[.025,.975])
    return dict(protocol=protocol,metric=metric,estimate=float(v[0]),ci_lower=float(lo),ci_upper=float(hi),bootstrap_draws=BOOT)

def main():
    started=time.time();status('RUNNING');hashes={}
    bi=json.loads((BENCH/'input_sha256.json').read_text());bo=json.loads((BENCH/'output_sha256.json').read_text())
    for p in [BASE/'counts.npy',BASE/'metadata.tsv',MAPPED/'out_of_fold_predictions.tsv',MAPPED/'fold_plan.json',BENCH/'out_of_fold_predictions.tsv',BENCH/'primary_summary.tsv',BENCH/'input_sha256.json',BENCH/'output_sha256.json',ROOT/'README.md',ROOT/'audit.py']:
        digest=sha(p);expected=bi.get(str(p),bo.get(str(p)))
        if expected is not None:assert digest==expected,str(p)
        hashes[str(p)]=digest
    save('input_sha256.json',hashes)
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    assert len(meta)==528 and meta.PATNO.is_unique and meta.label.sum()==358 and meta.batch.nunique()==51
    raw=np.load(BASE/'counts.npy',mmap_mode='r');assert raw.shape==(528,58780)
    totals=np.asarray(raw.sum(1,dtype=np.float64));assert (totals>0).all()
    meta['assigned_count_total']=totals;meta['log_library']=np.log(totals);meta['log_wbc']=np.log(meta.wbc)
    p=pd.read_csv(MAPPED/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str});p=p[p.penalty.eq('ridge')].copy()
    keys=['protocol','repeat','fold','PATNO']
    ref=pd.read_csv(BENCH/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str});ref=ref[ref.family.eq('rna')].set_index(keys).sort_index()
    actual=p.set_index(keys).sort_index();assert actual.index.equals(ref.index)
    np.testing.assert_allclose(actual[['probability_PD','threshold']],ref[['probability_PD','threshold']],atol=1e-8,rtol=0)
    assert len(p)==3168 and not p.duplicated(['protocol','repeat','PATNO']).any()
    assert p.probability_PD.between(0,1).all() and np.isfinite(p.threshold).all()
    plan=json.loads((MAPPED/'fold_plan.json').read_text());y=meta.label.to_numpy();p['training_prevalence']=np.nan
    primary=[]
    for s in plan:
        tr=np.array(s['train_indices']);te=np.array(s['test_indices'])
        assert not set(tr)&set(te) and set(tr)|set(te)==set(range(528))
        if s['protocol']=='batch_grouped':assert not set(meta.batch.iloc[tr])&set(meta.batch.iloc[te])
        take=(p.protocol==s['protocol'])&(p.repeat==s['repeat'])&(p.fold==s['fold'])
        d=p.loc[take].set_index('PATNO').loc[meta.PATNO.iloc[te]]
        assert len(d)==len(te) and np.array_equal(d.label,y[te])
        p.loc[take,'training_prevalence']=y[tr].mean()
        primary.append(dict(protocol=s['protocol'],repeat=s['repeat'],fold=s['fold'],AUROC=roc_auc_score(d.label,d.probability_PD),AP=average_precision_score(d.label,d.probability_PD)))
    assert p.training_prevalence.notna().all()
    primary=pd.DataFrame(primary);table('reproduced_primary_fold_metrics.tsv',primary)
    old=pd.read_csv(BENCH/'primary_summary.tsv',sep='\t');old=old[old.family.eq('rna')].set_index('protocol')
    for protocol in PROTOCOLS:
        np.testing.assert_allclose(primary.loc[primary.protocol.eq(protocol),['AUROC','AP']].mean(),old.loc[protocol,['AUROC_mean','AP_mean']].to_numpy(float),atol=1e-12,rtol=0)
    cov=['age_collection_years','RIN','intergenic_percent','log_library','log_wbc','neutrophils_percent','monocytes_percent','eosinophils_percent','basophils_percent']
    extra=['usable_percent','mapping_percent','coverage_bias','CBC_month_gap']
    assert meta[cov+extra].notna().all().all() and np.isfinite(meta[cov+extra]).all().all()
    design=pd.DataFrame(dict(const=np.ones(528),PD=y,male=meta.sex.eq('Male').astype(int),phase2=meta.phase.eq('PPMI-Phase2').astype(int)))
    scales=[]
    for col in cov:
        mu=meta[col].mean();sd=meta[col].std(ddof=0);assert sd>0
        design[col]=(meta[col]-mu)/sd;scales.append(dict(variable=col,mean=mu,SD=sd))
    assert np.linalg.matrix_rank(design)==design.shape[1]
    diagnostics=[dict(variable=col,VIF=float(variance_inflation_factor(design.to_numpy(),i))) for i,col in enumerate(design) if col!='const']
    table('joint_design_vif.tsv',diagnostics);table('covariate_scaling.tsv',scales)
    all_people=[];errors=[];per_repeat=[];summary=[];coefs=[];corr=[];bins=[];by_batch=[];by_phase=[];cal_validation=[];bootstrap_archive={};recurrence={}
    for protocol in PROTOCOLS:
        prob=[];threshold=[];baseline=[]
        for repeat in [1,2,3]:
            d=p[(p.protocol==protocol)&(p.repeat==repeat)].set_index('PATNO').loc[meta.PATNO]
            assert np.array_equal(d.label,y)
            prob.append(d.probability_PD.to_numpy());threshold.append(d.threshold.to_numpy());baseline.append(d.training_prevalence.to_numpy())
        prob=np.array(prob);threshold=np.array(threshold);baseline=np.array(baseline)
        sq=(prob-y)**2;logloss=-(y*np.log(np.clip(prob,1e-12,1))+(1-y)*np.log(np.clip(1-prob,1e-12,1)))
        person=meta.copy();person['protocol']=protocol;person['mean_probability_PD']=prob.mean(0);person['probability_SD']=prob.std(0)
        person['mean_Brier_loss']=sq.mean(0);person['mean_log_loss']=logloss.mean(0)
        person['mean_selected_margin']=(prob-threshold).mean(0)
        for mode,t in [('inner_selected',threshold),('fixed_0_5',.5)]:
            err=((prob>=t)!=y);nerr=err.sum(0);person['errors_'+mode]=nerr
            recurrence[(protocol,mode)]=nerr==3
            for label,group in [(0,'Control'),(1,'PD')]:
                for k in range(4):errors.append(dict(protocol=protocol,threshold=mode,diagnosis=group,errors_out_of_3=k,n=int(((nerr==k)&(y==label)).sum()),diagnosis_n=int((y==label).sum())))
        all_people.append(person)
        for repeat in range(3):
            for mode,t in [('inner_selected',threshold[repeat]),('fixed_0_5',.5)]:
                pred=prob[repeat]>=t;se=pred[y==1].mean();sp=(~pred[y==0]).mean()
                per_repeat.append(dict(protocol=protocol,repeat=repeat+1,threshold=mode,AUROC=roc_auc_score(y,prob[repeat]),AP=average_precision_score(y,prob[repeat]),Brier=sq[repeat].mean(),log_loss=logloss[repeat].mean(),sensitivity=se,specificity=sp,balanced_accuracy=(se+sp)/2,accuracy=(pred==y).mean(),mean_probability_PD=prob[repeat].mean()))
        w=weights(meta,protocol);ws=w.sum(1)
        for name,burden in [('Brier',sq.mean(0)),('log_loss',logloss.mean(0)),('Brier_training_prevalence',((baseline-y)**2).mean(0)),('Brier_minus_training_prevalence',(sq-(baseline-y)**2).mean(0)),('mean_prediction_minus_prevalence',prob.mean(0)-y)]:
            v=w@burden/ws;summary.append(interval(protocol,name,v));bootstrap_archive[protocol+'__'+name]=v
        summary.append(interval(protocol,'Brier_cohort_prevalence',w@((y.mean()-y)**2)/ws))
        joint=[];offset=[]
        for repeat in range(3):
            joint.append(calibration(y,prob[repeat],w));offset.append(calibration(y,prob[repeat],w,offset=True)[:,0])
            z=logit(np.clip(prob[repeat],1e-12,1-1e-12))
            for b in [0,1,19]:
                fit=sm.GLM(y,sm.add_constant(z),family=sm.families.Binomial(),freq_weights=w[b]).fit()
                fit0=sm.GLM(y,np.ones((528,1)),family=sm.families.Binomial(),offset=z,freq_weights=w[b]).fit()
                err=max(np.max(abs(fit.params-joint[-1][b])),abs(fit0.params[0]-offset[-1][b]))
                assert err<1e-6;cal_validation.append(dict(protocol=protocol,repeat=repeat+1,draw=b,max_parameter_error=float(err)))
        j=np.mean(joint,axis=0);o=np.mean(offset,axis=0)
        for name,v in [('calibration_joint_intercept',j[:,0]),('calibration_slope',j[:,1]),('calibration_offset_intercept',o)]:
            summary.append(interval(protocol,name,v));bootstrap_archive[protocol+'__'+name]=v
        table('calibration_per_repeat_'+protocol+'.tsv',[dict(repeat=i+1,joint_intercept=joint[i][0,0],slope=joint[i][0,1],offset_intercept=offset[i][0]) for i in range(3)])
        # Quantile-bin curve on participant mean predictions; fixed bins per bootstrap.
        q=pd.qcut(person.mean_probability_PD,10,labels=False,duplicates='drop').to_numpy()
        for k in np.unique(q):
            mask=q==k;den=w[:,mask].sum(1);val=np.divide(w[:,mask]@y[mask],den,out=np.full(len(w),np.nan),where=den>0);finite=np.isfinite(val[1:])
            lo,hi=np.quantile(val[1:][finite],[.025,.975]);bins.append(dict(protocol=protocol,bin=int(k+1),n=int(mask.sum()),n_PD=int(y[mask].sum()),predicted=prob.mean(0)[mask].mean(),observed=val[0],ci_lower=lo,ci_upper=hi,valid_bootstraps=int(finite.sum())))
        # Association regressions use ONE observation per person and cluster-robust inference.
        burden=person.mean_Brier_loss.to_numpy()
        def association(x,spec,terms):
            assert np.linalg.matrix_rank(x)==x.shape[1]
            model=sm.OLS(burden,x).fit(cov_type='cluster',cov_kwds={'groups':meta.batch,'use_correction':True,'df_correction':True},use_t=True)
            ci=model.conf_int()
            for term in terms:
                coefs.append(dict(protocol=protocol,specification=spec,variable=term,coefficient=float(model.params[term]),ci_lower=float(ci.loc[term,0]),ci_upper=float(ci.loc[term,1]),p_value=float(model.pvalues[term]),n=528,clusters=51,R_squared=float(model.rsquared),condition_number=float(np.linalg.cond(x))))
        association(design,'joint',list(design.columns[1:]))
        for col in extra:
            x=design.copy();x[col]=(meta[col]-meta[col].mean())/meta[col].std(ddof=0)
            association(x,'secondary_added_QC',[col])
        for label,group in [(0,'Control'),(1,'PD')]:
            mask=y==label
            for col in cov+extra+['lymphocytes_percent']:
                rho=spearmanr(meta.loc[mask,col],burden[mask]).statistic
                corr.append(dict(protocol=protocol,diagnosis=group,variable=col,n=int(mask.sum()),spearman_rho=float(rho)))
        for groupcol,out in [('batch',by_batch),('phase',by_phase)]:
            for group,d in person.groupby(groupcol):
                out.append(dict(protocol=protocol,group=group,n=len(d),n_PD=int(d.label.sum()),n_Control=int((1-d.label).sum()),mean_probability_PD=d.mean_probability_PD.mean(),Brier=d.mean_Brier_loss.mean(),mean_error_fraction=d.errors_inner_selected.mean()/3,recurring_errors=int((d.errors_inner_selected==3).sum())))
        print(protocol+' audit complete',flush=True)
    people=pd.concat(all_people,ignore_index=True);assert len(people)==1056 and not people.duplicated(['protocol','PATNO']).any()
    c=pd.DataFrame(coefs);assert c.p_value.notna().all();c['q_BH_all_association_tests']=multipletests(c.p_value,method='fdr_bh')[1]
    table('participant_error_audit.tsv',people);table('error_recurrence_counts.tsv',errors);table('pooled_metrics_per_repeat.tsv',per_repeat)
    table('calibration_and_loss_summary.tsv',summary);table('adjusted_error_associations.tsv',c);table('within_diagnosis_descriptive_correlations.tsv',corr)
    table('calibration_bins.tsv',bins);table('batch_error_summary.tsv',by_batch);table('phase_error_summary.tsv',by_phase);table('calibration_solver_validation.tsv',cal_validation)
    overlap=[]
    for mode in ['inner_selected','fixed_0_5']:
        a=recurrence[(PROTOCOLS[0],mode)];b=recurrence[(PROTOCOLS[1],mode)]
        for label,group in [(0,'Control'),(1,'PD')]:
            ids=(y==label);overlap.append(dict(threshold=mode,diagnosis=group,participant_protocol=int((a&ids).sum()),batch_protocol=int((b&ids).sum()),both=int((a&b&ids).sum()),either=int(((a|b)&ids).sum())))
    table('recurring_error_overlap.tsv',overlap)
    np.savez_compressed(ROOT/'bootstrap_calibration_loss.npz',**bootstrap_archive)
    report(meta,people,pd.DataFrame(errors),pd.DataFrame(summary),c,pd.DataFrame(per_repeat),pd.DataFrame(overlap),pd.DataFrame(bins),pd.DataFrame(corr),old)
    for path,digest in hashes.items():assert sha(path)==digest,path
    audit=dict(status='PASS',participants=528,PD=358,Control=170,batches=51,OOF_predictions=3168,participant_protocol_rows=1056,verified_primary_metrics=4,source_hashes_unchanged=len(hashes),association_tests=len(c),joint_design_condition_number=float(np.linalg.cond(design)),calibration_reference_max_error=max(r['max_parameter_error'] for r in cal_validation),bootstrap_draws=BOOT,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,statsmodels=sm.__version__,device='CPU, two numerical threads',classifier_refits=0)
    save('validation.json',audit)
    save('output_sha256.json',{str(p):sha(p) for p in ROOT.iterdir() if p.is_file() and p.name not in ['status.json','output_sha256.json','run.log']})
    status('COMPLETE',elapsed_seconds=round(time.time()-started,2),validation='PASS')

def report(meta,people,errors,s,c,metrics,overlap,bins,corr,old):
    labels={'participant_stratified':'Participant-stratified','batch_grouped':'Batch-grouped'}
    ss=s.set_index(['protocol','metric'])
    lines=['# Hallmark ridge prediction-error and calibration audit','','All 528 original participants; 358 PD and 170 controls; 51 sequencing batches. No classifier refitting, prediction correction, new exclusions or edits to earlier results.','','## Retained benchmark','','| Protocol | Primary AUROC | Primary AP |','|---|---:|---:|']
    for proto in PROTOCOLS:lines.append(f'| {labels[proto]} | {old.loc[proto,"AUROC_mean"]:.3f} | {old.loc[proto,"AP_mean"]:.3f} |')
    lines+=['','Primary benchmark metrics are mean within outer folds across repeats. All calibration and loss diagnostics below use pooled OOF predictions per repeat, then average repeats.','','## Calibration and probability error','','| Metric | Participant-stratified (95% interval) | Batch-grouped (95% interval) |','|---|---|---|']
    for metric in ['Brier','Brier_training_prevalence','Brier_minus_training_prevalence','log_loss','calibration_offset_intercept','calibration_slope','mean_prediction_minus_prevalence']:
        values=[]
        for proto in PROTOCOLS:
            r=ss.loc[(proto,metric)];values.append(f'{r.estimate:.3f} ({r.ci_lower:.3f} to {r.ci_upper:.3f})')
        lines.append('| '+metric+' | '+' | '.join(values)+' |')
    lines+=['','Smaller Brier/log loss is better. Calibration slope targets 1; offset-only intercept targets 0. A slope below 1 indicates overly extreme probabilities. Joint intercept and offset-only intercept are different parameters; both are saved. Calibration regressions were diagnostic only. No post-calibration performance is reported.','','## Recurrent classification errors','','Recurring means misclassified in all three repeats at each model’s original inner-selected threshold. These are three correlated assessments of the same participant.','','| Diagnosis | Participant protocol | Batch protocol | Both protocols |','|---|---:|---:|---:|']
    for r in overlap[overlap.threshold.eq('inner_selected')].itertuples():lines.append(f'| {r.diagnosis} | {r.participant_protocol} | {r.batch_protocol} | {r.both} |')
    lines+=['','All 0/3–3/3 counts at both thresholds are saved. Participant IDs and error records remain in the local participant_error_audit.tsv. No participant is excluded because of prediction error.','','| Protocol | Threshold | Sensitivity | Specificity | Balanced accuracy |','|---|---|---:|---:|---:|']
    for (proto,threshold),d in metrics.groupby(['protocol','threshold']):lines.append(f'| {labels[proto]} | {threshold} | {d.sensitivity.mean():.3f} | {d.specificity.mean():.3f} | {d.balanced_accuracy.mean():.3f} |')
    sig=c[c.q_BH_all_association_tests<.05]
    lines+=['','## Technical and blood-cell associations','','Outcome: each participant’s mean OOF Brier loss across repeats. Joint models include diagnosis, sex, collection age, RIN, intergenic percentage, log assigned library total, phase, log WBC and four differential-cell percentages. All continuous coefficients are per one SD; binary coefficients compare 1 versus 0. Positive coefficients mean greater prediction error. Inference uses 51-batch cluster-robust t intervals. BH adjustment covers all '+str(len(c))+' reported tests.']
    if len(sig):
        lines+=['','| Protocol | Specification | Variable | Brier change (95% interval) | BH q |','|---|---|---|---|---:|']
        for r in sig.itertuples():lines.append(f'| {labels[r.protocol]} | {r.specification} | {r.variable} | {r.coefficient:+.4f} ({r.ci_lower:+.4f} to {r.ci_upper:+.4f}) | {r.q_BH_all_association_tests:.4g} |')
    else:lines+=['','No reported association passed BH q<0.05. This does not establish absence of technical or cell-composition effects.']
    lines+=['','All coefficients, including nonsignificant ones, are in adjusted_error_associations.tsv. Diagnosis-specific correlations are descriptive, without significance claims. Phase/batch tables retain sample sizes and diagnosis counts; small-batch variation is not evidence of a technical cause.','','## Interpretation and next decision','','This audit identifies associations and probability reliability; it cannot establish that correcting an associated factor will improve AUROC. If probabilities are too extreme, a subsequent training-only calibration comparison could test Brier/log-loss improvement. Monotonic calibration preserves ranking within a fixed score set and should not be proposed as an AUROC remedy. Any technical correction must be learned inside training folds and tested against the unchanged benchmark. Do not delete difficult participants or optimize thresholds on these OOF outcomes.','','## Scope of uncertainty','','Intervals use 2,000 fixed-prediction bootstrap draws shared across repeats, with diagnosis-stratified participant resampling or whole-batch resampling. They omit RNA-model retraining and adaptive-search variation. Batch-robust regression accounts for within-batch residual correlation but not every dependence caused by overlapping CV training sets. The analyses are exploratory. Calibration pertains to this selected case/control sample, whose PD prevalence is 67.8%; it is not population PD risk. The repeat-averaged probability plot is descriptive and is not a new model-performance estimate. CBCs were measured in the same or preceding two months.','','## Artifacts','','- participant_error_audit.tsv: one person per protocol, metadata, probability and loss summaries, recurrence counts.','- adjusted_error_associations.tsv and joint_design_vif.tsv: complete estimates, multiplicity and collinearity checks.','- within_diagnosis_descriptive_correlations.tsv, phase_error_summary.tsv and batch_error_summary.tsv.','- calibration_and_loss_summary.tsv, calibration_per_repeat_*.tsv, calibration_bins.tsv and bootstrap archive.','- validation.json and SHA256 manifests: provenance, participant and numerical checks.','','![Audit composite](audit_composite.png)','','Method: [Van Calster et al., 2019](https://doi.org/10.1186/s12916-019-1466-7). See README.md for the analysis specification.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    figures(people,errors,s,c,bins,labels)

def figures(people,errors,summary,c,bins,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,10));colors=['#247BA0','#D97732']
    ax=axes[0,0]
    for proto,color in zip(PROTOCOLS,colors):
        d=bins[bins.protocol.eq(proto)];ax.plot(d.predicted,d.observed,'o-',label=labels[proto],color=color)
        ax.vlines(d.predicted,d.ci_lower,d.ci_upper,color=color,alpha=.4)
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean OOF predicted PD probability',ylabel='Observed PD fraction',title='Calibration of repeat-averaged predictions');ax.legend(fontsize=8,loc='upper left')
    ax=axes[0,1]
    for i,(proto,color) in enumerate(zip(PROTOCOLS,colors)):
        d=errors[(errors.protocol==proto)&(errors.threshold=='inner_selected')].groupby('errors_out_of_3').n.sum().reindex(range(4))
        ax.bar(np.arange(4)+(i-.5)*.36,d.values,width=.34,color=color,label=labels[proto])
        for k,n in enumerate(d.values):ax.text(k+(i-.5)*.36,n+3,str(n),ha='center',fontsize=9)
    ax.set(xticks=range(4),xlabel='Errors across three repeats',ylabel='Participants',title='Errors at training-selected thresholds');ax.legend(fontsize=8)
    ax=axes[1,0];terms=['RIN','intergenic_percent','log_library','phase2','log_wbc','neutrophils_percent','monocytes_percent','eosinophils_percent','basophils_percent']
    pretty=['RIN','Intergenic reads','Log library total','Phase 2 vs 1','Log WBC','Neutrophils','Monocytes','Eosinophils','Basophils']
    for i,(proto,color) in enumerate(zip(PROTOCOLS,colors)):
        d=c[(c.protocol==proto)&(c.specification=='joint')].set_index('variable').loc[terms];yy=np.arange(len(terms))+(i-.5)*.23
        ax.hlines(yy,d.ci_lower,d.ci_upper,color=color);ax.scatter(d.coefficient,yy,color=color,s=22)
    ax.axvline(0,color='gray',ls='--');ax.set(yticks=np.arange(len(terms)),yticklabels=pretty,xlabel='Adjusted change in mean Brier loss',title='Technical and cell associations');ax.invert_yaxis()
    ax=axes[1,1]
    for i,(proto,color) in enumerate(zip(PROTOCOLS,colors)):
        for j,metric in enumerate(['Brier','Brier_training_prevalence']):
            r=summary[(summary.protocol==proto)&(summary.metric==metric)].iloc[0];x=j+(i-.5)*.20
            ax.vlines(x,r.ci_lower,r.ci_upper,color=color,lw=2);ax.scatter(x,r.estimate,color=color,s=45)
    ax.set(xticks=[0,1],xticklabels=['Hallmark ridge','Training-prevalence\nconstant'],ylabel='Mean OOF Brier score',title='Probability error (lower is better)')
    for letter,ax in zip('ABCD',axes.flat):ax.text(-.12,1.06,letter,transform=ax.transAxes,fontsize=14,fontweight='bold')
    fig.suptitle('Hallmark ridge: prediction-error and calibration audit',fontsize=16,y=.98)
    fig.subplots_adjust(left=.12,right=.97,top=.90,bottom=.14,wspace=.34,hspace=.40)
    fig.text(.12,.035,'528 participants • Three repeats • Original OOF predictions and thresholds\nBlue: participant-stratified; orange: batch-grouped. Bars: conditional 95% intervals.\nPanel C: continuous covariates per SD; binary phase contrast; joint adjustment with batch-cluster inference.',fontsize=9,linespacing=1.5)
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/('audit_composite.'+ext),dpi=600,facecolor='white')
    fig.savefig(ROOT/'audit_composite_preview.png',dpi=110,facecolor='white');plt.close(fig)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except BaseException as e:
        status('FAILED',error=repr(e));traceback.print_exc();raise
