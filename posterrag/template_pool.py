"""RAP-25K template pool: the retrieval index comes from annotations/all.jsonl; templates and posters are fetched on
demand from the Hugging Face dataset (one ~450 MB tar shard per 100 templates), so the full 110 GB is never needed.

    pool = TemplatePool()                         # downloads annotations/all.jsonl (~80 MB) once
    row = pool.by_id['ICLR_2025_5006']
    html = pool.fetch_template(row)               # -> path to <paper_id>/generated.html (assets/ next to it)
    img = pool.fetch_poster(row)                  # -> path to the poster image

A local copy of the dataset (same layout) can be used with TemplatePool(local_root='/path/to/RAP').
"""
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path

REPO_ID = 'VincentHancoder/RAP'


class TemplatePool:
    def __init__(self, repo_id=REPO_ID, local_root=None, cache_dir=None, annotations=None):
        self.repo_id = repo_id
        self.local_root = Path(local_root) if local_root else None
        self.cache_dir = Path(cache_dir or os.environ.get('RAP_CACHE', Path.home() / '.cache' / 'rap'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = annotations or self._get('annotations/all.jsonl')
        self.rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
        self.by_id = {r['paper_id']: r for r in self.rows}
        self.index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # domain -> task -> method -> [idx]
        for i, r in enumerate(self.rows):
            self.index[r.get('domain', '')][r.get('task', '')][r.get('method', '')].append(i)
        self.taxonomy = {k: sorted({r.get(k, '') for r in self.rows if r.get(k)}) for k in ('domain', 'task', 'method')}

    # ------------------------------------------------------------------ files
    def _get(self, rel):
        if self.local_root is not None:
            p = self.local_root / rel
            if p.exists(): return str(p)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo_id, rel, repo_type='dataset', cache_dir=str(self.cache_dir / 'hf'))

    def fetch_poster(self, row):
        return self._get(row['poster_file'])

    def fetch_template(self, row):
        """Return the path of <paper_id>/generated.html, extracting it from its shard if needed."""
        pid = row['paper_id']
        if self.local_root is not None and (self.local_root / 'templates' / pid / 'generated.html').exists():
            return str(self.local_root / 'templates' / pid / 'generated.html')
        out = self.cache_dir / 'templates' / pid
        if not (out / 'generated.html').exists():
            tar_path = self._get(row['template_archive']); prefix = row.get('template_dir', pid + '/')
            with tarfile.open(tar_path) as t:
                members = [m for m in t.getmembers() if m.name.startswith(prefix)]
                t.extractall(out.parent, members=members)
        return str(out / 'generated.html')

    # -------------------------------------------------------------- retrieval
    def exact_match(self, domain, task, method):
        return list(self.index.get(domain, {}).get(task, {}).get(method, []))
