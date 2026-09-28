"""Run both frozen timing strata with four CPU workers per DESeq2 process."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib
import json
import subprocess
import shutil
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
RSCRIPT = shutil.which('Rscript')
assert RSCRIPT, 'Rscript must be available on PATH'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def one(stratum):
    assert json.loads((ROOT/stratum/'preflight.json').read_text())['n'] in [112, 416]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (ROOT/stratum/'run.log').open('w') as stream:
        result = subprocess.run([RSCRIPT, str(ROOT/'timing_models.R'), stratum],
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0, f'{stratum} failed: inspect run.log'
    return stratum


if __name__ == '__main__':
    start = time.monotonic()
    paths = [WORK/'ppmi-deseq2-2026-09-12/primary/dds.rds',
             WORK/'ppmi-deseq2-2026-09-12/msigdb_hallmark_gobp.rds',
             WORK/'ppmi-blood-cell-adjustment-2026-09-12/blood_covariates.tsv',
             ROOT/'README.md', ROOT/'timing_models.R', ROOT/'run_timing.py',
             ROOT/'finalize_enrichment.R', ROOT/'finalize_timing.py']
    hashes = {str(p):sha(p) for p in paths}
    (ROOT/'timing_input_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (ROOT/'timing_status.json').write_text(json.dumps(dict(status='RUNNING', strata=['same_month', 'preceding_months'])))
    try:
        for stratum in ['same_month', 'preceding_months']:
            subprocess.run([RSCRIPT, str(ROOT/'timing_models.R'), stratum, '--prepare-only'], check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            completed = list(pool.map(one, ['same_month', 'preceding_months']))
        subprocess.run([RSCRIPT, str(ROOT/'finalize_enrichment.R'), *completed], check=True)
        assert all(sha(Path(p)) == value for p, value in hashes.items())
        (ROOT/'timing_status.json').write_text(json.dumps(dict(status='COMPLETE', strata=completed,
              elapsed_seconds=time.monotonic()-start, source_hashes_unchanged=True), indent=2)+'\n')
        print((ROOT/'timing_status.json').read_text(), flush=True)
    except Exception as error:
        (ROOT/'timing_status.json').write_text(json.dumps(dict(status='FAILED', error=repr(error)), indent=2)+'\n')
        raise
