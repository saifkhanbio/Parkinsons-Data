from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,time,hashlib,traceback
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit,expit
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from calibrator import fit,apply
AUDIT=ROOT.parent/'ppmi-ridge-error-audit-2026-09-15'
BENCH=ROOT.parent/'ppmi-hallmark-blood-block-ridge-2026-09-15'
MAPPED=ROOT.parent/'ppmi-mapped-gene-classifier-2026-09-12'
BASE=ROOT.parent/'ppmi-classifier-2026-09-12'
sys.path.insert(0,str(AUDIT))
from audit import weights,calibration,interval,PROTOCOLS,BOOT
# The reference module declares its own cache path; restore this run's path
# before importing plotting code. Importing it does not create any files.
os.environ['MPLCONFIGDIR']=str(ROOT/'matplotlib_cache')
sys.path.insert(0,str(MAPPED))
from mapped_genes import select_threshold
KEYS=['repeat','protocol','fold']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def table(name,data):pd.DataFrame(data).to_csv(ROOT/name,sep='\t',index=False)
def status(stage,**kw):save('status.json',dict(status=stage,**kw));print(stage,flush=True)

def main():
    start=time.time();status('RUNNING');hashes={}
    bo=json.loads((BENCH/'output_sha256.json').read_text());bi=json.loads((BENCH/'input_sha256.json').read_text())
    ao=json.loads((AUDIT/'output_sha256.json').read_text())
    sources=[BENCH/name for name in ['inner_oof_probabilities.npz','out_of_fold_predictions.tsv','fold_plan.json','selected_parameters.tsv','primary_summary.tsv','output_sha256.json','input_sha256.json']]
    sources += [BASE/'metadata.tsv',MAPPED/'out_of_fold_predictions.tsv',MAPPED/'fold_plan.json',MAPPED/'mapped_genes.py',ROOT.parent/'ppmi-classifier-refinement-retry-2026-09-12/refine.py',ROOT.parent/'ppmi-classifier-refinement-retry-2026-09-12/elastic_solver.py',BASE/'classifier.py',AUDIT/'audit.py',AUDIT/'calibration_and_loss_summary.tsv']
    sources += list(ROOT.glob('*.py'))+[ROOT/'README.md']
    for p in sources:
        d=sha(p);expected=bo.get(str(p),bi.get(str(p),ao.get(str(p))))
        if expected is not None:assert d==expected,str(p)
        hashes[str(p)]=d
    save('input_sha256.json',hashes)
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str});y=meta.label.to_numpy()
    assert len(meta)==528 and meta.PATNO.is_unique and sum(y)==358 and meta.batch.nunique()==51
    plan=json.loads((BENCH/'fold_plan.json').read_text());assert plan==json.loads((MAPPED/'fold_plan.json').read_text())
    save('fold_plan.json',plan)
    params=pd.read_csv(BENCH/'selected_parameters.tsv',sep='\t').query("family=='rna'").set_index(KEYS)
    pred=pd.read_csv(BENCH/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str}).query("family=='rna'").copy()
    original=pd.read_csv(MAPPED/'out_of_fold_predictions.tsv',sep='\t',dtype={'PATNO':str}).query("penalty=='ridge'")
    a=pred.set_index(KEYS+['PATNO']).sort_index();b=original.set_index(KEYS+['PATNO']).sort_index();assert a.index.equals(b.index)
    np.testing.assert_allclose(a[['probability_PD','threshold']],b[['probability_PD','threshold']],atol=1e-8,rtol=0)
    cache=np.load(BENCH/'inner_oof_probabilities.npz');out=[];fits=[];foldmetrics=[];checks=[]
    # Synthetic optimum/reference, serialization and lower-bound tests.
    rng=np.random.default_rng(451);sp=expit(rng.normal(0,2,240));sy=rng.binomial(1,expit(.4+.35*logit(sp)))
    ss=fit(sp,sy);rr=sm.GLM(sy,sm.add_constant(logit(sp)),family=sm.families.Binomial()).fit()
    np.testing.assert_allclose([ss['intercept'],ss['slope']],rr.params,atol=1e-5)
    np.testing.assert_array_equal(apply(ss,sp),apply(json.loads(json.dumps(ss)),sp))
    reverse=fit(np.linspace(.01,.99,100),np.r_[np.ones(50),np.zeros(50)]);assert reverse['slope_at_bound']
    seen={}
    for n,s in enumerate(plan,1):
        context={k:s[k] for k in KEYS};key=tuple(s[k] for k in KEYS);tr=np.array(s['train_indices']);te=np.array(s['test_indices'])
        assert not set(tr)&set(te) and set(tr)|set(te)==set(range(528))
        if s['protocol']=='batch_grouped':assert not set(meta.batch.iloc[tr])&set(meta.batch.iloc[te])
        seen.setdefault((s['protocol'],s['repeat']),np.zeros(528,int))[te]+=1
        np.testing.assert_array_equal(cache[f'{n}_train_indices'],tr)
        chosen=params.loc[key];cp=cache[f'{n}_rna_{float(chosen.C)}_None'];assert cp.shape==(len(tr),) and np.isfinite(cp).all()
        iv=np.zeros(len(tr),int);aucs=[]
        for f in s['inner']:
            train=np.array(f['train_positions']);val=np.array(f['validation_positions'])
            assert not set(train)&set(val) and set(train)|set(val)==set(range(len(tr)))
            assert set(y[tr[train]])==set(y[tr[val]])=={0,1}
            if s['protocol']=='batch_grouped':assert not set(meta.batch.iloc[tr[train]])&set(meta.batch.iloc[tr[val]])
            iv[val]+=1;aucs.append(roc_auc_score(y[tr[val]],cp[val]))
        assert (iv==1).all()
        np.testing.assert_allclose(aucs,np.fromstring(chosen.inner_AUROCs,sep=';'),atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.mean(aucs),chosen.mean_inner_AUROC,atol=1e-12,rtol=0)
        t,_=select_threshold(y[tr],cp);np.testing.assert_allclose(t,chosen.threshold,atol=1e-8,rtol=0)
        state=fit(cp,y[tr]);state.update(**context,outer_fold_number=n,C=float(chosen.C),training_indices=tr.tolist(),training_PATNO=meta.PATNO.iloc[tr].tolist(),cache_key=f'{n}_rna_{float(chosen.C)}_None')
        # Perturb every outer test label; they cannot affect the fitted mapping.
        altered=y.copy();altered[te]=1-altered[te];again=fit(cp,altered[tr])
        assert again['intercept']==state['intercept'] and again['slope']==state['slope']
        reference_error=0.
        if not state['slope_at_bound']:
            ref=sm.GLM(y[tr],sm.add_constant(logit(cp)),family=sm.families.Binomial()).fit()
            reference_error=float(np.max(abs(ref.params-[state['intercept'],state['slope']])))
            assert reference_error<1e-5
        # Only after fitting the map retrieve the outer-held-out scores/labels.
        old=pred[(pred.protocol==s['protocol'])&(pred.repeat==s['repeat'])&(pred.fold==s['fold'])].set_index('PATNO').loc[meta.PATNO.iloc[te]]
        assert len(old)==len(te) and np.array_equal(old.label,y[te])
        p=old.probability_PD.to_numpy();q=apply(state,p);mapped_threshold=float(apply(state,np.array([t]))[0])
        np.testing.assert_array_equal(np.argsort(p,kind='stable'),np.argsort(q,kind='stable'))
        np.testing.assert_array_equal(p>=t,q>=mapped_threshold)
        np.testing.assert_array_equal(q,apply(json.loads(json.dumps(state)),p))
        # Applying arbitrary held-out scores leaves mapping state unchanged.
        state_before=json.dumps(state,sort_keys=True);apply(state,np.linspace(.01,.99,len(te)));assert state_before==json.dumps(state,sort_keys=True)
        (ROOT/'fold_calibrators').mkdir(exist_ok=True);save(f'fold_calibrators/fold_{n:02d}.json',state)
        fits.append({k:v for k,v in state.items() if k not in ['training_indices','training_PATNO']})
        for family,pr,th in [('original',p,t),('calibrated',q,mapped_threshold)]:
            auc=roc_auc_score(y[te],pr);ap=average_precision_score(y[te],pr)
            foldmetrics.append(dict(**context,family=family,AUROC=auc,AP=ap,Brier=float(np.mean((pr-y[te])**2))))
            out.extend(dict(**context,family=family,PATNO=meta.PATNO.iloc[i],label=int(y[i]),probability_PD=float(v),threshold=th) for i,v in zip(te,pr))
        np.testing.assert_allclose([roc_auc_score(y[te],p),average_precision_score(y[te],p)],[roc_auc_score(y[te],q),average_precision_score(y[te],q)],atol=1e-12,rtol=0)
        checks.append(dict(**context,status='PASS',training_only=True,inner_cache_reproduced=True,rank_preserved=True,selected_decisions_preserved=True,reference_error=reference_error))
    assert len(seen)==6 and all((v==1).all() for v in seen.values())
    out=pd.DataFrame(out);assert len(out)==6336 and not out.duplicated(['protocol','repeat','family','PATNO']).any()
    table('out_of_fold_predictions.tsv',out);table('calibration_parameters.tsv',fits);table('outer_fold_metrics.tsv',foldmetrics);table('fold_validation.tsv',checks)
    status('REPORTING');report(meta,out,pd.DataFrame(foldmetrics))
    for p,h in hashes.items():assert sha(p)==h,p
    save('validation.json',dict(status='PASS',participants=528,outer_folds=30,calibrators=30,classifier_refits=0,all_source_hashes_unchanged=True,inner_cache_checks=30,heldout_perturbation_checks=30,rank_and_decision_checks=30,maximum_reference_parameter_error=max(x['reference_error'] for x in checks),positive_slope_boundary_fits=sum(x['slope_at_bound'] for x in fits),bootstrap_draws=BOOT))
    save('output_sha256.json',{str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and 'matplotlib_cache' not in p.parts and p.name not in ['run.log','status.json','output_sha256.json','.launch.lock']})
    status('COMPLETE',elapsed_seconds=round(time.time()-start,2))

def report(meta,pred,folds):
    y=meta.label.to_numpy();summary=[];differences=[];repeatmetrics=[];bins=[];archive={}
    oldsummary=pd.read_csv(AUDIT/'calibration_and_loss_summary.tsv',sep='\t').set_index(['protocol','metric'])
    benchmark=pd.read_csv(BENCH/'primary_summary.tsv',sep='\t').query("family=='rna'").set_index('protocol')
    for proto in PROTOCOLS:
        w=weights(meta,proto);ws=w.sum(1);draws={}
        for family in ['original','calibrated']:
            arrays=[];offset=[];joint=[]
            for repeat in [1,2,3]:
                d=pred[(pred.protocol==proto)&(pred.family==family)&(pred.repeat==repeat)].set_index('PATNO').loc[meta.PATNO];assert np.array_equal(d.label,y)
                p=d.probability_PD.to_numpy();arrays.append(p)
                joint.append(calibration(y,p,w));offset.append(calibration(y,p,w,offset=True)[:,0])
                for mode,t in [('fixed_0_5',.5),('transformed_inner_selected',d.threshold.to_numpy())]:
                    positive=p>=t;se=positive[y==1].mean();sp=(~positive[y==0]).mean()
                    repeatmetrics.append(dict(protocol=proto,family=family,repeat=repeat,threshold_mode=mode,pooled_AUROC=roc_auc_score(y,p),pooled_AP=average_precision_score(y,p),sensitivity=se,specificity=sp,balanced_accuracy=(se+sp)/2,Brier=np.mean((p-y)**2),log_loss=np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p)))))
            pp=np.array(arrays);jj=np.mean(joint,axis=0)
            values={'Brier':w@np.mean((pp-y)**2,axis=0)/ws,'log_loss':w@np.mean(-(y*np.log(pp)+(1-y)*np.log1p(-pp)),axis=0)/ws,'calibration_slope':jj[:,1],'calibration_offset_intercept':np.mean(offset,axis=0),'mean_prediction_minus_prevalence':w@(pp.mean(0)-y)/ws}
            for metric,v in values.items():
                row=interval(proto,metric,v);row['family']=family;summary.append(row);draws[(family,metric)]=v;archive[proto+'__'+family+'__'+metric]=v
                if family=='original':np.testing.assert_allclose([row['estimate'],row['ci_lower'],row['ci_upper']],oldsummary.loc[(proto,metric),['estimate','ci_lower','ci_upper']].to_numpy(float),atol=1e-8,rtol=0)
            meanp=pp.mean(0);q=pd.qcut(meanp,10,labels=False,duplicates='drop')
            for k in np.unique(q):
                mask=q==k;den=w[:,mask].sum(1);obs=np.divide(w[:,mask]@y[mask],den,out=np.full(len(w),np.nan),where=den>0);lo,hi=np.nanquantile(obs[1:],[.025,.975])
                bins.append(dict(protocol=proto,family=family,bin=int(k+1),n=int(mask.sum()),predicted=meanp[mask].mean(),observed=obs[0],ci_lower=lo,ci_upper=hi))
        for metric in values:
            v=draws[('calibrated',metric)]-draws[('original',metric)];differences.append(interval(proto,metric,v))
        for family in ['original','calibrated']:
            d=folds[(folds.protocol==proto)&(folds.family==family)]
            np.testing.assert_allclose(d[['AUROC','AP']].mean().to_numpy(),benchmark.loc[proto,['AUROC_mean','AP_mean']].to_numpy(float),atol=1e-12,rtol=0)
    s=pd.DataFrame(summary);delta=pd.DataFrame(differences);fm=folds.groupby(['protocol','family'],as_index=False)[['AUROC','AP']].mean()
    table('probability_quality_summary.tsv',s);table('paired_changes.tsv',delta);table('primary_discrimination.tsv',fm);table('secondary_pooled_metrics.tsv',repeatmetrics);table('calibration_bins.tsv',bins)
    np.savez_compressed(ROOT/'bootstrap_probabilities_metrics.npz',**archive)
    lines=['# Training-only calibration comparison','','Original Hallmark gene ridge versus positive-slope logistic calibration of its saved scores. Same 528 participants and 30 outer folds. Thirty calibrators were fitted exclusively to the matching outer-training inner OOF predictions and labels; no RNA models were refitted.','','## Held-out probability quality','','Lower Brier and log loss are better. Intervals use 2,000 paired conditional bootstrap draws.','','| Protocol | Metric | Original (95% interval) | Calibrated (95% interval) | Change (95% interval) |','|---|---|---|---|---|']
    for proto in PROTOCOLS:
        for metric in ['Brier','log_loss','calibration_slope','calibration_offset_intercept']:
            a=s[(s.protocol==proto)&(s.metric==metric)].set_index('family');d=delta[(delta.protocol==proto)&(delta.metric==metric)].iloc[0]
            fmt=lambda r:f'{r.estimate:.3f} ({r.ci_lower:.3f} to {r.ci_upper:.3f})'
            lines.append(f'| {proto} | {metric} | {fmt(a.loc["original"])} | {fmt(a.loc["calibrated"])} | {fmt(d)} |')
    lines+=['','## Discrimination and thresholds','','Mean-within-fold AUROC and AP were unchanged in every outer fold. Fold-specific transforms can change pooled OOF rankings across folds; pooled AUROC/AP are secondary and any change does not establish improved within-model discrimination.','','| Protocol | Original/calibrated AUROC | Original/calibrated AP |','|---|---:|---:|']
    for proto in PROTOCOLS:
        a=fm[(fm.protocol==proto)&(fm.family=='original')].iloc[0];lines.append(f'| {proto} | {a.AUROC:.3f} | {a.AP:.3f} |')
    lines+=['','Transforming the original inner-selected threshold preserved every original binary decision. Results at probability 0.5 are separately saved and were not used to choose the method. The diagnostic calibration fits on outer predictions were never applied to those predictions.','','## Interpretation','','This is a probability-reliability comparison, not an AUROC improvement experiment. Brier and log loss assess probability quality and are not pure measures of calibration. Inspect calibration slopes/intercepts and curves alongside them. A positive result would support retaining a calibrated version for this cohort alongside the unchanged gene-level ranking benchmark; it would not establish population PD risk or clinical utility.','','The inner predictions used for calibration were also used to select ridge C. Outer folds evaluate that entire fixed procedure, but the calibrator training fit is not independent of C selection. Inner RNA models use smaller training sets than their outer refits, which may affect calibration transfer. Prior cohort-wide model comparisons and the error audit motivated this exploratory analysis. Conditional uncertainty holds predictions fixed and omits retraining and adaptive-search variation.','','## Saved outputs','','Thirty JSON fold calibrators, source hashes, selected-C/cache checks, held-out perturbation checks, rank/decision preservation, optimizer/reference checks, predictions, bootstrap arrays and figures are retained. There is no single final deployment calibrator: no outer test labels were used to fit one. Previous analyses and manuscript were unchanged.','','![Calibration comparison](calibration_comparison.png)','','Methods: [probability calibration guidance](https://scikit-learn.org/stable/modules/calibration.html); [Van Calster et al.](https://doi.org/10.1186/s12916-019-1466-7).','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines));figure(s,pd.DataFrame(bins))

def figure(summary,bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,9));colors=['#6B7280','#177F70']
    for col,proto in enumerate(PROTOCOLS):
        ax=axes[0,col]
        for f,c in zip(['original','calibrated'],colors):
            d=bins[(bins.protocol==proto)&(bins.family==f)];ax.plot(d.predicted,d.observed,'o-',color=c,label=f.capitalize());ax.vlines(d.predicted,d.ci_lower,d.ci_upper,color=c,alpha=.35)
        ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean OOF probability of PD',ylabel='Observed PD fraction',title=proto.replace('_',' ').capitalize());ax.legend()
    for col,metric in enumerate(['Brier','log_loss']):
        ax=axes[1,col]
        for i,proto in enumerate(PROTOCOLS):
            for j,(f,c) in enumerate(zip(['original','calibrated'],colors)):
                r=summary[(summary.protocol==proto)&(summary.family==f)&(summary.metric==metric)].iloc[0];x=i+(j-.5)*.22
                ax.vlines(x,r.ci_lower,r.ci_upper,color=c,lw=2);ax.scatter(x,r.estimate,color=c,s=50)
        ax.set(xticks=[0,1],xticklabels=['Participant folds','Batch folds'],ylabel=metric.replace('_',' ').capitalize(),title='Held-out '+metric.replace('_',' ')+' (lower is better)')
    for l,ax in zip('ABCD',axes.flat):ax.text(-.12,1.06,l,transform=ax.transAxes,fontweight='bold',fontsize=14)
    fig.suptitle('Training-only calibration of Hallmark ridge probabilities',fontsize=15,y=.98)
    fig.subplots_adjust(left=.09,right=.97,top=.89,bottom=.14,hspace=.4,wspace=.3)
    fig.text(.09,.035,'528 participants • Original outer folds • Three repeats • No RNA refitting\nGray: original; green: calibrated. Bars: conditional 95% intervals. Curves use repeat-averaged probabilities.\nEach calibration mapping was learned inside its outer training set; within-fold AUROC/AP and mapped-threshold decisions are preserved.',fontsize=8.5,linespacing=1.5)
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/('calibration_comparison.'+ext),dpi=600,facecolor='white')
    fig.savefig(ROOT/'calibration_comparison_preview.png',dpi=110,facecolor='white');plt.close(fig)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except BaseException as e:status('FAILED',error=repr(e));traceback.print_exc();raise
