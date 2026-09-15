"""Freeze combined-stratum holdouts and matched demographic control plans."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'ppmi-classifier-2026-09-12'
SEED=20260912

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def prepare():
    assert not (ROOT/'plan.json').exists(),'Plan already exists'
    meta=pd.read_csv(BASE/'metadata.tsv',sep='\t',dtype={'PATNO':str})
    assert len(meta)==528 and meta.PATNO.is_unique and meta.label.value_counts().to_dict()=={1:358,0:170}
    age=meta.enrollment_age
    meta['age_band']=np.select([age.between(30,50,inclusive='both'),(age>50)&(age<=70),(age>70)&(age<=80),age>80],
                      ['30_50','gt50_70','gt70_80','gt80'],default='unclassified')
    meta['stratum']=meta.sex+'_'+meta.age_band
    meta['allocation']='unclassified_age'
    combined=[];counts=[]
    for sex in ['Female','Male']:
        for band in ['30_50','gt50_70','gt70_80','gt80']:
            name=sex+'_'+band;idx=np.flatnonzero(meta.stratum.eq(name));y=meta.label.to_numpy()[idx]
            n=np.bincount(y,minlength=2);row=dict(stratum=name,sex=sex,age_band=band,n=len(idx),PD=int(n[1]),Control=int(n[0]))
            if min(n)<4:
                row.update(eligible=False,reason='Insufficient class counts for held-out testing plus inner DE training with >=2 per class')
                meta.loc[idx,'allocation']='insufficient_stratum_counts';counts.append(row);continue
            tr,te=train_test_split(idx,test_size=.2,stratify=y,random_state=SEED)
            tr,te=sorted(map(int,tr)),sorted(map(int,te));train_n=np.bincount(meta.label.iloc[tr],minlength=2)
            assert min(train_n)>=3 and len(set(meta.label.iloc[te]))==2
            row.update(eligible=True,reason='',n_train=len(tr),n_test=len(te),test_PD=int(meta.label.iloc[te].sum()),test_Control=int((meta.label.iloc[te]==0).sum()))
            row['sparse_test']=min(row['test_PD'],row['test_Control'])<10
            counts.append(row);combined.append(dict(name=name,sex=sex,age_band=band,train_indices=tr,test_indices=te))
            meta.loc[tr,'allocation']='training';meta.loc[te,'allocation']='held_out'
    train=sorted(i for s in combined for i in s['train_indices']);test=sorted(i for s in combined for i in s['test_indices'])
    assert len(train)==410 and len(test)==105 and not set(train)&set(test)
    models=[]
    for s in combined:models.append(dict(name='combined_'+s['name'],family='combined',**{k:s[k] for k in ['sex','age_band','train_indices','test_indices']}))
    for sex in ['Female','Male']:
        models.append(dict(name='sex_'+sex,family='sex_only',sex=sex,
          train_indices=[i for i in train if meta.sex.iloc[i]==sex],test_indices=[i for i in test if meta.sex.iloc[i]==sex]))
    for band in ['30_50','gt50_70','gt70_80']:
        models.append(dict(name='age_'+band,family='age_only',age_band=band,
          train_indices=[i for i in train if meta.age_band.iloc[i]==band],test_indices=[i for i in test if meta.age_band.iloc[i]==band]))
    models.append(dict(name='pooled_de',family='pooled_de',train_indices=train,test_indices=test))
    selectors={};lookup={}
    def selector(idx,label):
        idx=sorted(map(int,idx));key=tuple(idx)
        if key in lookup:return lookup[key]
        name=label;lookup[key]=name
        selectors[name]=dict(name=name,train_indices=idx,training_PATNO=meta.PATNO.iloc[idx].tolist(),design='~ group')
        return name
    for m in models:
        tr=np.array(m['train_indices']);y=meta.label.iloc[tr].to_numpy();n_inner=min(10,int(np.bincount(y).min()))
        assert not set(tr)&set(test)
        splits=StratifiedKFold(n_splits=n_inner,shuffle=True,random_state=SEED+1).split(np.zeros(len(tr)),y)
        m['inner']=[]
        for j,(a,b) in enumerate(splits,1):
            assert np.bincount(y[a],minlength=2).min()>=2 and len(set(y[b]))==2
            m['inner'].append(dict(train_positions=a.tolist(),validation_positions=b.tolist(),selector=selector(tr[a],m['name']+f'_inner{j}')))
        m['outer_selector']=selector(tr,m['name']+'_outer')
    control=dict(models[-1]);control.update(name='pooled_hallmark',family='pooled_hallmark');models.append(control)
    assert len(models)==13
    meta.to_csv(ROOT/'participant_allocation.tsv',sep='\t',index=False)
    pd.DataFrame(counts).to_csv(ROOT/'stratum_counts.tsv',sep='\t',index=False)
    save('plan.json',dict(seed=SEED,combined=combined,models=models,train_indices=train,test_indices=test,selectors=list(selectors.values())))
    save('planning_summary.json',dict(participants=528,allocated=515,training=410,held_out=105,unallocated=13,feasible_combined_strata=len(combined),models=len(models),DE_fits=len(selectors),new_R_design='~ group'))
    genes=np.load(BASE/'gene_ids.npy');assert len(genes)==58780
    (ROOT/'gene_ids.txt').write_text('\n'.join(genes)+'\n')
    save('planning_source_sha256.json',{str(p):sha(p) for p in [BASE/'metadata.tsv',BASE/'counts.npy',BASE/'gene_ids.npy',ROOT/'README.md',ROOT/'prepare_plan.py']})
    print((ROOT/'planning_summary.json').read_text())

if __name__=='__main__':prepare()
