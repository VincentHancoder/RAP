"""PosterRAG: training-free, retrieval-augmented paper-to-poster generation over the RAP-25K template pool.

    export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY (+OPENAI_BASE_URL) / GEMINI_API_KEY
    python -m posterrag.run_posterrag --pdf paper.pdf --out outputs/my_poster
    python -m posterrag.run_posterrag --pdf paper.pdf --out outputs/my_poster --backend openai --preferences "figure-heavy, dense"
    python -m posterrag.run_posterrag --pdf paper.pdf --out outputs/my_poster --template ICLR_2025_5006   # skip retrieval

Stage 1 (template retrieval): ParserAgent -> RetrievalAgent-1 categorical filtering -> RetrievalAgent-2 narrative
matching -> RetrievalAgent-3 aesthetic selection. Stage 2 (poster generation): MappingAgent (slot analysis, content
mapping, programmatic filling) -> ReviewerAgent (see-then-fix). Templates come from the public RAP-25K dataset and are
downloaded on demand (one ~450 MB shard per 100 templates); pass --local-root for a local copy of the dataset.
Outputs: <out>/{parsed_content.json, pmeta.json, retrieval.json, mapping/, review/, poster.html, poster.png, summary.json}.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from .llm import LLM
from .template_pool import TemplatePool
from .agents.parser import parse_paper, layoutlmv3_formula_detector
from .agents.retrieval import retrieve
from .agents.mapping import run_mapping
from .agents.reviewer import review_and_fix, screenshot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdf', required=True, help='paper PDF, or a LaTeX project .zip')
    ap.add_argument('--out', required=True)
    ap.add_argument('--backend', default=os.environ.get('RAP_LLM_BACKEND', 'anthropic'), choices=['anthropic', 'openai', 'gemini'])
    ap.add_argument('--model', default=None, help='override the backend default model')
    ap.add_argument('--preferences', default=None, help='free-text style preferences for RetrievalAgent-3 (default: neutral)')
    ap.add_argument('--template', default=None, help='paper_id of a template to use directly (skips retrieval)')
    ap.add_argument('--exclude', nargs='*', default=[], help='paper_ids never to retrieve (e.g. the paper itself)')
    ap.add_argument('--local-root', default=None, help='local copy of the RAP-25K dataset (annotations/, templates/, posters/)')
    ap.add_argument('--max-review-rounds', type=int, default=3); ap.add_argument('--review-score-threshold', type=int, default=8)
    ap.add_argument('--formulas', action='store_true', help='also crop formulas with the LayoutLMv3 detector (needs the layout environment, see inference/README.md)')
    a = ap.parse_args()

    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    llm = LLM(a.backend, a.model); print(f'[PosterRAG] backend {a.backend} / {llm.model}')
    pool = TemplatePool(local_root=a.local_root)
    print(f'[TemplatePool] {len(pool.rows)} templates, {len(pool.taxonomy["domain"])} domains')

    # Stage 1 -------------------------------------------------------------------------------------------------
    pmeta = parse_paper(a.pdf, out / 'parse', llm, pool.taxonomy, layoutlmv3_formula_detector if a.formulas else None)
    if a.template:
        row = pool.by_id[a.template]; retrieval = {'template': row, 'level': 'manual'}
    else:
        retrieval = retrieve(pool, pmeta, llm, a.preferences, exclude_ids=a.exclude)
        row = retrieval['template']
    (out / 'retrieval.json').write_text(json.dumps({k: v for k, v in retrieval.items()}, indent=2, ensure_ascii=False), encoding='utf-8')
    template_html = pool.fetch_template(row); print(f'[TemplatePool] template {row["paper_id"]}: {template_html}')

    # Stage 2 -------------------------------------------------------------------------------------------------
    html, template, mapping = run_mapping(pmeta, template_html, llm, out / 'mapping')
    html, review = review_and_fix(html, template, llm, out / 'review', a.max_review_rounds, a.review_score_threshold)
    final = out / 'poster.html'; final.write_text(html, encoding='utf-8')
    assets = Path(template_html).parent / 'assets'
    if assets.is_dir(): shutil.copytree(assets, out / 'assets', dirs_exist_ok=True)
    try: screenshot(final, out / 'poster.png', template['canvas_width'], template['canvas_height'])
    except Exception as e:                                                                           # noqa: BLE001
        print(f'  final render failed: {e}')
    summary = {'paper': str(a.pdf), 'template': row['paper_id'], 'retrieval_level': retrieval.get('level'),
               'labels': pmeta['labels'], 'mapping_stats': mapping['stats'], 'review': {k: review[k] for k in ('final_score', 'fixes_applied')},
               'canvas': [template['canvas_width'], template['canvas_height']], 'seconds': round(time.time() - t0, 1)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[PosterRAG] done in {summary['seconds']} s -> {final}")


if __name__ == '__main__':
    main()
