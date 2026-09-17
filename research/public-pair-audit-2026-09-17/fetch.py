"""Download a pinned, small public reference corpus; no model weights or calls."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

DOCS = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, default=Path('private/public-pair-audit-2026-09-17'))
ROOT = parser.parse_args().workspace.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
lock = json.loads((DOCS / 'source-lock.json').read_text())
tokenizer = json.loads((DOCS / 'tokenizer-lock.json').read_text())
jobs = [{'path': 'upstream/' + f['path'], 'bytes': f['bytes'], 'sha256': f['sha256'],
         'url': f'https://raw.githubusercontent.com/{lock["repository"]}/{lock["commit"]}/{f["path"]}'}
        for f in lock['records']]
jobs.append({'path': 'gpt2-tokenizer.json', **{k: tokenizer[k] for k in ['url', 'bytes', 'sha256']}})

def fetch(job):
    destination = ROOT / job['path']
    if destination.exists():
        raw = destination.read_bytes()
    else:
        with urllib.request.urlopen(job['url'], timeout=25) as r:
            raw = r.read(job['bytes'] + 1)
    if len(raw) != job['bytes'] or hashlib.sha256(raw).hexdigest() != job['sha256']:
        raise ValueError('Source integrity mismatch: ' + job['path'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return len(raw)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    total = sum(pool.map(fetch, jobs))
for name in ['source.json', 'tokenizer-lock.json']:
    (ROOT / name).write_bytes((DOCS / name).read_bytes())
print(f'Checked {len(jobs)} files, {total} bytes. No model weights downloaded.')
