import numpy as np

def check_plan(plan,meta):
    train=set(plan['train_indices']);test=set(plan['test_indices'])
    assert len(train)==410 and len(test)==105 and not train&test
    registry={e['name']:e for e in plan['selectors']}
    def selector(name,idx):
        e=registry[name];assert e['train_indices']==sorted(map(int,idx))
        assert e['training_PATNO']==meta.PATNO.iloc[e['train_indices']].tolist()
        assert not set(idx)&test
    for model in plan['models']:
        tr=np.array(model['train_indices']);te=np.array(model['test_indices'])
        assert set(tr)<=train and set(te)<=test and len(set(tr))==len(tr) and len(set(te))==len(te)
        assert set(meta.label.iloc[te])=={0,1}
        seen=[]
        for fold in model['inner']:
            a=tr[fold['train_positions']];b=tr[fold['validation_positions']]
            assert not set(a)&set(b) and set(a)|set(b)==set(tr)
            assert meta.label.iloc[a].value_counts().min()>=2 and set(meta.label.iloc[b])=={0,1}
            selector(fold['selector'],a);seen.extend(b)
        assert sorted(seen)==sorted(tr)
        selector(model['outer_selector'],tr)
    for family in ['combined','sex_only','age_only','pooled_de','pooled_hallmark']:
        indices=[i for m in plan['models'] if m['family']==family for i in m['test_indices']]
        assert sorted(indices)==sorted(test)
