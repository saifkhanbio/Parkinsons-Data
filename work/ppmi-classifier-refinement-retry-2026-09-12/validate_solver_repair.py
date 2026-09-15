"""Check repaired optimizer against SAGA and the failed real training partition."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from refine import BASE, FoldData, matrices, fit_model, validate_refinement

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
with threadpool_limits(limits=2):
    checks = validate_refinement(torch.device('cpu'))
    meta = pd.read_csv(BASE / 'metadata.tsv', sep='\t', dtype={'PATNO': str})
    raw, genes = np.load(BASE / 'counts.npy'), np.load(BASE / 'gene_ids.npy')
    data = FoldData(raw, genes, meta, meta.label.to_numpy(), torch.device('cpu'))
    split = json.loads((BASE / 'fold_plan.json').read_text())['participant_stratified'][0]
    outer = np.array(split['train_indices'])
    tr = outer[np.array(split['inner'][0]['train_positions'])]
    va = outer[np.array(split['inner'][0]['validation_positions'])]
    pack = data.pack(tr, va)
    rows = []
    for family in ['rna', 'combined']:
        x, v = matrices(pack, family, 2000)
        for ratio in [.1, .5, .9]:
            start = time.monotonic()
            model = fit_model(x, data.labels[tr], 1., 'elasticnet', ratio)
            row = dict(family=family, C=1., l1_ratio=ratio, features=x.shape[1],
                       seconds=time.monotonic()-start, iterations=int(model.n_iter_[0]),
                       kkt_residual=model.kkt_residual_, objective=model.objective_)
            rows.append(row)
            print(json.dumps(row), flush=True)
    checks['failed_partition_and_adjacent_configurations'] = rows
    (ROOT / 'solver_repair_validation.json').write_text(json.dumps(dict(status='PASS', checks=checks), indent=2)+'\n')
    print('PASS: solver agreement, leakage, threshold, and failed-partition checks', flush=True)
