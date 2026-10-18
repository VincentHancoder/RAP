"""Generate posters with RAP-Qwen3.8-27B-RL.

Two modes:
  * benchmark  - run the exact prompts of the RAP-150 test set (benchmark/rap_test150/prompts.jsonl)
  * pdf        - parse a new paper PDF with the training-aligned pipeline and generate a poster for it

Examples
  python -m inference.generate benchmark --out outputs/rap_test150
  python -m inference.generate pdf --pdf paper.pdf --out outputs/my_poster \
      --instruction "Create a clear academic poster ..." --canvas "Landscape · 3:2" --columns 3 --accent "#584287"
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODEL_ID = os.environ.get('RAP_MODEL', 'VincentHancoder/RAP-Qwen3.8-27B-RL')
WINDOW = 16384          # training context: input + output tokens


def load_model(model_id=MODEL_ID, dtype=torch.bfloat16, device_map='cuda'):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map=device_map,
                                                         attn_implementation='sdpa').eval()
    return model, processor


def messages_from_benchmark(row, base):
    """benchmark/rap_test150/prompts.jsonl stores the user turn as text with <image> markers (ms-swift style);
    rebuild the multimodal chat the processor expects, interleaving the thumbnails in order."""
    system, user = row['messages'][0]['content'], row['messages'][1]['content']
    parts = user.split('<image>'); images = [str(Path(base) / p) for p in row['images']]
    assert len(parts) == len(images) + 1, f"{row['paper_id']}: {len(parts) - 1} <image> markers vs {len(images)} images"
    content = []
    for k, text in enumerate(parts):
        if text: content.append({'type': 'text', 'text': text})
        if k < len(images): content.append({'type': 'image', 'image': images[k]})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': system}]}, {'role': 'user', 'content': content}]


@torch.inference_mode()
def generate(model, processor, messages, max_new_tokens=8192, max_time=None):
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                           return_dict=True, return_tensors='pt').to(model.device)
    n_in = int(inputs['input_ids'].shape[1])
    budget = min(max_new_tokens, WINDOW - n_in)
    if budget <= 0:
        raise ValueError(f'prompt is {n_in} tokens, over the {WINDOW}-token window')
    t0 = time.monotonic()
    out = model.generate(**inputs, max_new_tokens=budget, max_time=max_time, do_sample=False, use_cache=True,
                         temperature=None, top_p=None, top_k=None, repetition_penalty=1.1)
    new = out[0, n_in:]
    text = processor.decode(new, skip_special_tokens=True)
    return text, {'input_tokens': n_in, 'output_tokens': int(new.shape[0]), 'seconds': round(time.monotonic() - t0, 1),
                  'complete_html': bool(re.search(r'</html\s*>', text, re.I))}


def run_benchmark(args):
    from poster_render import safe_html
    bench = Path(args.benchmark); rows = [json.loads(l) for l in open(bench / 'prompts.jsonl', encoding='utf-8')]
    if args.ids: rows = [r for r in rows if r['paper_id'] in set(args.ids)]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model)
    assets_root = Path(args.assets) if args.assets else None
    for i, row in enumerate(rows):
        pid = row['paper_id']; d = out / pid; d.mkdir(exist_ok=True)
        if (d / 'poster.html').exists() and not args.overwrite:
            print(f'[{i + 1}/{len(rows)}] {pid}: exists, skip', flush=True); continue
        text, info = generate(model, processor, messages_from_benchmark(row, bench), args.max_new_tokens)
        (d / 'model_output.txt').write_text(text, encoding='utf-8')
        info.update(paper_id=pid, title=row.get('title'))
        if assets_root is not None:                                   # full-resolution crops -> self-contained HTML + PNG
            assets = [{'id': p.stem, 'file': str(p)} for p in sorted((assets_root / pid / 'assets').glob('*.png'))]
            try:
                html, used, missing = safe_html(text, assets); (d / 'poster.html').write_text(html, encoding='utf-8')
                info.update(assets_used=used, assets_missing=missing)
                if not args.no_render:
                    from render_html import render_png; render_png(d / 'poster.html', d / 'poster.png')
            except Exception as e:                                    # noqa: BLE001
                info['render_error'] = str(e)[:300]
        (d / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
        print(f"[{i + 1}/{len(rows)}] {pid}: {info['output_tokens']} tokens in {info['seconds']} s"
              f"{'' if info['complete_html'] else ' (incomplete)'}", flush=True)


def run_pdf(args):
    from pdf_pipeline import prepare, chat, DEFAULT_INSTRUCTION
    from poster_render import render
    job = Path(args.out); job.mkdir(parents=True, exist_ok=True)
    p = prepare(args.pdf, job, args.instruction or DEFAULT_INSTRUCTION, args.canvas, args.accent, args.columns, args.spec or '')
    model, processor = load_model(args.model)
    text, info = generate(model, processor, chat(p), args.max_new_tokens)
    png, html, archive, meta = render(text, p)
    info.update(meta); (job / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"poster: {png}\nhtml:   {html}\n{info['output_tokens']} tokens in {info['seconds']} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('benchmark', help='run the RAP-150 test prompts')
    b.add_argument('--benchmark', default=str(ROOT.parent / 'benchmark' / 'rap_test150'))
    b.add_argument('--assets', default=None, help='directory with <paper_id>/assets/*.png full-resolution crops (see README)')
    b.add_argument('--ids', nargs='*', default=None); b.add_argument('--overwrite', action='store_true'); b.add_argument('--no-render', action='store_true')
    p = sub.add_parser('pdf', help='generate a poster for a new paper PDF')
    p.add_argument('--pdf', required=True); p.add_argument('--instruction', default=None); p.add_argument('--spec', default=None)
    p.add_argument('--canvas', default='Landscape · 3:2'); p.add_argument('--columns', type=int, default=3); p.add_argument('--accent', default='#584287')
    for s in (b, p):
        s.add_argument('--out', required=True); s.add_argument('--model', default=MODEL_ID); s.add_argument('--max-new-tokens', type=int, default=12288)
    args = ap.parse_args()
    (run_benchmark if args.mode == 'benchmark' else run_pdf)(args)


if __name__ == '__main__':
    main()
