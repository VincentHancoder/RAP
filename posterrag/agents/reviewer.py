"""ReviewerAgent: see-then-fix loop. Each round renders the poster with headless Chromium, asks the VLM for a score and
block-level CSS patches, and applies them under safety constraints that block (1) visibility-hiding rules, (2) text-hostile
overflow/truncation, (3) layout-geometry changes and (4) structural / colour-scheme changes. Stops at score >= threshold,
after max_rounds, or when no patch is proposed. Adapted from Paper2Poster's visual reviewer (MIT)."""
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote

from ..llm import extract_json
from .parser import render_prompt

HIDING = {'display': {'none'}, 'visibility': {'hidden', 'collapse'}, 'opacity': {'0', '0.0'}}
TEXT_HOSTILE = {'overflow': {'hidden', 'clip'}, 'overflow-x': {'hidden', 'clip'}, 'overflow-y': {'hidden', 'clip'},
                'white-space': {'nowrap', 'pre'}, 'text-overflow': {'ellipsis', 'clip'}}
GEOMETRY = {'left', 'top', 'right', 'bottom', 'width', 'height', 'position', 'transform', 'margin', 'margin-left', 'margin-top'}
SCHEME = {'background', 'background-color', 'background-image', 'border-color'}


def screenshot(html_path, png_path, width, height, timeout_ms=120000):
    from playwright.sync_api import sync_playwright
    exe = os.environ.get('RAP_CHROMIUM') or shutil.which('chromium') or shutil.which('chromium-browser') or None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=exe, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': width, 'height': height})
        page.goto('file://' + quote(Path(html_path).resolve().as_posix(), safe='/:'), wait_until='load', timeout=timeout_ms)
        page.wait_for_timeout(1500); page.screenshot(path=str(png_path), full_page=True, timeout=timeout_ms); browser.close()
    return str(png_path)


def _patch_css(css, bid, prop, value):
    selector = f'#{bid} img' if prop.startswith('img::') else f'#{bid} .content'; prop = prop.split('::')[-1]
    pat = re.compile(rf'({re.escape(selector)}\s*\{{)(.*?)(\}})', re.DOTALL); m = pat.search(css)
    if not m: return css + f'\n{selector} {{\n  {prop}: {value};\n}}\n'
    body = m.group(2); pp = re.compile(rf'(\s*{re.escape(prop)}\s*:\s*)([^;]*)(;?)', re.DOTALL)
    body = pp.sub(rf'\g<1>{value};', body, count=1) if pp.search(body) else body.rstrip() + (';' if body.strip() and not body.rstrip().endswith(';') else '') + f'\n  {prop}: {value};'
    return pat.sub(lambda mm: mm.group(1) + body + '\n' + mm.group(3), css, count=1)


def apply_fixes(html, fixes, template):
    """Apply block-level CSS patches, dropping anything in the four blocked categories (and font-size decreases)."""
    m = re.search(r'(<style[^>]*>)(.*?)(</style>)', html, re.DOTALL)
    if not m: return html, 0
    css = m.group(2); fonts = {s['block_id']: s['font_size'] for s in template['slots']}; types = {s['block_id']: s['block_type'] for s in template['slots']}; applied = 0
    for fix in fixes:
        bid = fix.get('block_id', '')
        if bid not in fonts or fix.get('truncate_text'): continue
        for prop, value in (fix.get('css_fixes') or {}).items():
            real, v = prop.split('::')[-1].strip().lower(), str(value).strip().lower()
            if real in HIDING and v in HIDING[real]: continue
            if types[bid] == 'text-like' and real in TEXT_HOSTILE and v in TEXT_HOSTILE[real]: continue
            if real in GEOMETRY or real in SCHEME: continue
            if real == 'font-size':
                n = re.match(r'(\d+)', v)
                if n and int(n.group(1)) < fonts[bid]: continue
            css = _patch_css(css, bid, prop, str(value)); applied += 1
    return html[:m.start(2)] + css + html[m.end(2):], applied


def review_and_fix(html, template, llm, out_dir, max_rounds=3, score_threshold=8):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); w, h = template['canvas_width'], template['canvas_height']
    from .mapping import slots_summary
    summary = slots_summary(template, include_text=False); rounds, previous, total_fixes, score = [], [], 0, 0
    print('[ReviewerAgent]')
    for r in range(1, max_rounds + 1):
        html_path = out / f'review_r{r}.html'; html_path.write_text(html, encoding='utf-8')
        try: png = screenshot(html_path, out / f'review_r{r}.png', w, h)
        except Exception as e:                                                                       # noqa: BLE001
            print(f'  screenshot failed: {e}'); break
        system, prompt = render_prompt('visual_reviewer_agent', round_num=r, max_rounds=max_rounds, canvas_width=w, canvas_height=h,
                                       slots_summary=summary, previous_fixes=json.dumps(previous, ensure_ascii=False) if previous else '')
        resp = llm.vision(prompt, [png], system=system, temperature=0.2, max_tokens=16384)
        try: review = extract_json(resp['content'])
        except Exception as e:                                                                       # noqa: BLE001
            print(f'  unparseable review: {e}'); break
        score, fixes = int(review.get('overall_score', 0) or 0), review.get('fixes', []) or []
        rounds.append({'round': r, 'score': score, 'fixes': len(fixes), 'notes': review.get('review_notes', ''), 'tokens': {'input': resp['input_tokens'], 'output': resp['output_tokens']}})
        print(f'  round {r}: score {score}/10, {len(fixes)} fixes')
        if score >= score_threshold or not fixes: break
        html, n = apply_fixes(html, fixes, template); total_fixes += n; previous.extend(fixes)
    (out / 'review_log.json').write_text(json.dumps({'rounds': rounds, 'fixes_applied': total_fixes, 'final_score': score}, indent=2, ensure_ascii=False), encoding='utf-8')
    return html, {'rounds': rounds, 'fixes_applied': total_fixes, 'final_score': score}
