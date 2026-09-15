"""Check public source syntax and file scope without loading or executing analyses."""
from pathlib import Path
import ast
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.cache', '.venv', 'venv', 'dependencies', 'R-library'}


def main():
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True)
    if tracked.returncode == 0:
        paths = [ROOT / p for p in tracked.stdout.decode().split('\0') if p]
    else:
        paths = [p for p in ROOT.rglob('*') if p.is_file() and not IGNORED.intersection(p.relative_to(ROOT).parts)]
    counts = {'Python': 0, 'R': 0, 'shell': 0}
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if rel == 'Supplementary material.zip':
            continue  # Existing author-uploaded archive, outside the new code payload.
        assert p.suffix in {'.py', '.R', '.sh', '.md'} or rel in {
            '.gitignore', 'requirements.txt', 'requirements-boosting.txt', 'config/private_config.example.json'
        }, f'Unexpected publication file: {rel}'
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=rel)
            compile(p.read_text(), rel, 'exec')
            counts['Python'] += 1
        elif p.suffix == '.R':
            counts['R'] += 1
            if shutil.which('Rscript'):
                subprocess.run(['Rscript', '-e', 'invisible(parse(file=commandArgs(TRUE)[1]))', str(p)], check=True, capture_output=True)
        elif p.suffix == '.sh':
            counts['shell'] += 1
            subprocess.run(['bash', '-n', str(p)], check=True)
    config = json.loads((ROOT / 'config/private_config.example.json').read_text())
    assert all(v == [] for v in config.values()), 'Public configuration must not contain participant identifiers'
    print('Source checks passed:', counts)
    if not shutil.which('Rscript'):
        print('Rscript unavailable: R parsing was not checked on this machine.')
    print('No study code was executed and no participant data were loaded.')


if __name__ == '__main__':
    main()
