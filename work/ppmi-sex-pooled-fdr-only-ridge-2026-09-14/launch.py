"""Launch once, detached; status is checked only on user request."""
from pathlib import Path
import os,sys,json,subprocess,fcntl
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    lock=(ROOT/'.launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'job.json').exists(),'Job already launched; inspect before restarting'
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    with (ROOT/'run.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,'-B',str(ROOT/'worker.py')],cwd=ROOT.parents[1],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job=dict(pid=worker.pid,worker=str(ROOT/'worker.py'),status_file=str(ROOT/'status.json'),polling='Only on user request')
    (ROOT/'job.json').write_text(json.dumps(job,indent=2)+'\n');print(json.dumps(job),flush=True)
