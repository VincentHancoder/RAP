"""MappingAgent: slot analysis -> LLM content mapping (replace_text / replace_image / replace_rendered / keep / clear)
-> deterministic programmatic filling. Layout, typography, colours and decorations of the template are never touched.
Slot parsing and filling are adapted from Paper2Poster's template-filling pipeline (MIT, see LICENSE-Paper2Poster)."""
import json
import re
import shutil
from pathlib import Path

from ..llm import extract_json
from .parser import render_prompt

# ------------------------------------------------------------------------------------------------ slot analysis
_TEXT_BLOCK = re.compile(r'<div\s+id="(blk-\d+)"\s+class="block\s+(text-like[^"]*)"\s+data-label="([^"]*)"\s+data-position="(\d+)"\s+style="([^"]*)"[^>]*>(.*?)</div>\s*</div>', re.DOTALL)
_IMAGE_BLOCK = re.compile(r'<div\s+id="(blk-\d+)"\s+class="block\s+(image-like[^"]*)"\s+data-label="([^"]*)"\s+data-position="(\d+)"\s+style="([^"]*)"[^>]*>(.*?)</div>', re.DOTALL)


def _css_of(css, block_id):
    m = re.search(rf'#{block_id}\s+\.content\s*\{{([^}}]*)\}}', css, re.DOTALL); props = {}
    if m:
        for line in m.group(1).split(';'):
            if ':' in line: k, v = line.split(':', 1); props[k.strip()] = v.strip()
    return props


def text_capacity(width, height, font_size):
    """approximate characters that fit: floor(w / (f*0.55)) lines of floor(h / (f*1.4))"""
    font_size = max(font_size, 1); cpl = max(1, int(width / (font_size * 0.55))); lines = max(1, int(height / (font_size * 1.4)))
    return {'chars_per_line': cpl, 'max_lines': lines, 'approx_max_chars': cpl * lines}


def parse_template(html_path):
    html = Path(html_path).read_text(encoding='utf-8')
    css = '\n'.join(b.strip() for b in re.findall(r'<style[^>]*>(.*?)</style>', html, re.DOTALL))
    cw, ch = 1200, 900
    m = re.search(r'(?:html|body|#canvas)\s*\{[^}]*width:\s*(\d+)px[^}]*height:\s*(\d+)px', html, re.DOTALL)
    if m: cw, ch = int(m.group(1)), int(m.group(2))
    slots = []
    for m in list(_TEXT_BLOCK.finditer(html)) + list(_IMAGE_BLOCK.finditer(html)):
        bid, cls, label, pos, style, inner = m.groups()
        st = dict(p.strip().split(':', 1) for p in style.split(';') if ':' in p); px = lambda k: int(float(st.get(k, '0').strip().replace('px', '') or 0))
        props = _css_of(css, bid); fs = int((re.match(r'(\d+)', props.get('font-size', '')) or [None, '20'])[1] or 20)
        s = {'block_id': bid, 'block_type': 'text-like' if 'text-like' in cls else 'image-like', 'data_label': label, 'data_position': int(pos),
             'left': px('left'), 'top': px('top'), 'width': px('width'), 'height': px('height'), 'font_size': fs, 'css_props': props,
             'content_text': '', 'img_src': ''}
        if s['block_type'] == 'text-like':
            c = re.search(r'<div class="content"[^>]*>(.*)', inner, re.DOTALL); s['content_text'] = (c.group(1).strip() if c else '')
            s['text_capacity'] = text_capacity(s['width'], s['height'], fs)
        else:
            i = re.search(r'src="([^"]*)"', inner); s['img_src'] = i.group(1) if i else ''
        slots.append(s)
    slots.sort(key=lambda s: s['data_position'])
    for i, a in enumerate(slots):                                   # near-duplicate slots (IoU >= .8) are cleared, keeping the first
        for b in slots[i + 1:]:
            if b.get('_hidden_overlap'): continue
            ix = max(0, min(a['left'] + a['width'], b['left'] + b['width']) - max(a['left'], b['left']))
            iy = max(0, min(a['top'] + a['height'], b['top'] + b['height']) - max(a['top'], b['top']))
            inter = ix * iy; union = a['width'] * a['height'] + b['width'] * b['height'] - inter
            if union > 0 and inter / union >= 0.8: b['_hidden_overlap'] = True
    return {'html_text': html, 'canvas_width': cw, 'canvas_height': ch, 'css_full': css, 'slots': slots}


def slots_summary(template, include_text=True):
    out = [f"Template: {template['canvas_width']}x{template['canvas_height']}px", f"Total slots: {len(template['slots'])}", '']
    for s in template['slots']:
        dims = f"{s['width']}x{s['height']}px at ({s['left']},{s['top']})"
        if s['block_type'] == 'text-like':
            c = s['text_capacity']; out.append(f"[{s['data_position']}] {s['block_id']} | text-like | label={s['data_label']} | {dims} | font={s['font_size']}px | capacity=~{c['max_lines']} lines x {c['chars_per_line']} chars")
            if include_text and s['content_text']: out.append(f'     current text: "{s["content_text"][:100]}{"..." if len(s["content_text"]) > 100 else ""}"')
        else: out.append(f"[{s['data_position']}] {s['block_id']} | image-like | label={s['data_label']} | {dims} | src={s['img_src']}")
    return '\n'.join(out)


# ------------------------------------------------------------------------------------- content mapping (LLM)
def map_content(pmeta, template, llm):
    p = pmeta['parsed']
    system, prompt = render_prompt('content_mapper_agent', meta=p['meta'], abstract=p.get('abstract', ''), sections=p['sections'], figures=p['figures'],
                                   tables=p['tables'], slots_summary=slots_summary(template), rendered_equations_tables=list(pmeta['rendered_registry']),
                                   raw_sections=p.get('_raw_sections', []))
    r = llm.text(prompt, system=system, temperature=0.3, max_tokens=32768)
    mappings = extract_json(r['content']).get('slot_mappings', [])
    return _backfill(mappings, template, p), {'input': r['input_tokens'], 'output': r['output_tokens']}


def _backfill(mappings, template, parsed):
    """A text slot the LLM left empty gets paper text instead (every text slot must carry content)."""
    text_slots = {s['block_id']: s for s in template['slots'] if s['block_type'] == 'text-like'}
    source = parsed.get('_raw_sections') or parsed['sections']; k = 0
    for m in mappings:
        if m.get('action') == 'clear' and m.get('block_id') in text_slots and source:
            cap = text_slots[m['block_id']]['text_capacity']['approx_max_chars']; sec = source[k % len(source)]; k += 1
            m.update(action='replace_text', new_text=(sec.get('content', '') if cap > 80 else sec.get('title', ''))[:cap])
    return mappings


# ------------------------------------------------------------------------------------ programmatic filling
def _set_text(html, bid, text):
    pat = re.compile(r'(<div\s+id="' + re.escape(bid) + r'"[^>]*>)\s*(<div\s+class="content"[^>]*>)(.*?)</div>\s*</div>', re.DOTALL)
    return pat.sub(lambda m: m.group(1) + m.group(2) + text + '</div></div>', html, count=1)


def _set_src(html, bid, src):
    pat = re.compile(r'(<div\s+id="' + re.escape(bid) + r'"[^>]*>)\s*<img\s+([^>]*?)src="[^"]*"([^>]*)/?>', re.DOTALL)
    return pat.sub(lambda m: f'{m.group(1)}<img {m.group(2)}src="{src}"{m.group(3)}/>', html, count=1)


def _hide(html, bid):
    pat = re.compile(r'(<div\s+id="' + re.escape(bid) + r'"[^>]*?)(\sstyle=")', re.DOTALL)
    new = pat.sub(lambda m: m.group(1) + ' data-hidden="1" style="display:none;', html, count=1)
    return new if new != html else re.sub(r'(<div\s+id="' + re.escape(bid) + r'")', r'\1 style="display:none"', html, count=1)


def fill_template(template, mappings, pmeta):
    html = template['html_text']; figs = pmeta['parsed']['figures']; freg, rreg = pmeta['figure_registry'], pmeta['rendered_registry']
    hidden = {s['block_id'] for s in template['slots'] if s.get('_hidden_overlap')}
    stats = dict(replaced_text=0, replaced_image=0, replaced_rendered=0, kept=0, cleared=0, errors=0); used_f, used_r = set(), set()
    for m in mappings:
        bid, act = m.get('block_id', ''), m.get('action', 'keep')
        if bid in hidden: html = _hide(_set_text(html, bid, ''), bid); stats['cleared'] += 1; continue
        if act == 'replace_text':
            new = _set_text(html, bid, m.get('new_text', ''))
            if new != html: html = new; stats['replaced_text'] += 1
            else: stats['errors'] += 1
        elif act == 'replace_image':
            fid = m.get('figure_id'); uri = None
            if fid is not None and fid not in used_f:
                for f in figs:
                    if str(f['figure_id']) == str(fid):
                        for g in f.get('graphics_paths', []):
                            key = next((k for k in (g, Path(g).stem, f'figure_{fid}') if k in freg), None)
                            if key: uri = freg[key]['base64_uri']; break
            if uri:
                new = _set_src(html, bid, uri)
                if new != html: html = new; used_f.add(fid); stats['replaced_image'] += 1; continue
            html = _hide(html, bid); stats['cleared'] += 1
        elif act == 'replace_rendered':
            key = m.get('rendered_key', '')
            if key in rreg and key not in used_r:
                new = _set_src(html, bid, rreg[key])
                if new == html: new = _set_text(html, bid, f'<img src="{rreg[key]}" style="max-width:100%;max-height:100%;object-fit:contain;"/>')
                html = new; used_r.add(key); stats['replaced_rendered'] += 1
            else: html = _hide(html, bid); stats['cleared'] += 1
        elif act == 'clear': html = _hide(_set_text(html, bid, ''), bid); stats['cleared'] += 1
        else: stats['kept'] += 1
    mapped = {m.get('block_id') for m in mappings}
    for s in template['slots']:                                   # image slots the mapper did not mention still show the source paper's assets
        if s['block_type'] == 'image-like' and s['block_id'] not in mapped: html = _hide(html, s['block_id']); stats['cleared'] += 1
    for bid in re.findall(r'<div\s+id="(blk-\d+)"[^>]*>\s*<img[^>]*src="assets/[^"]*"', html):
        if 'display:none' not in re.search(r'<div\s+id="' + bid + r'"[^>]*>', html).group(0): html = _hide(html, bid); stats['cleared'] += 1
    title = pmeta['parsed']['meta'].get('poster_title', '')
    if title: html = re.sub(r'<title>.*?</title>', f'<title>{title}</title>', html, count=1, flags=re.DOTALL)
    baseline = '\n.block.text-like .content{word-wrap:break-word;overflow-wrap:break-word;white-space:normal}\n.block.image-like img{max-width:100%;max-height:100%;object-fit:contain}\n'
    html = re.sub(r'(</style>)', baseline + r'\1', html, count=1)
    print(f'  fill: {stats}')
    return html, stats


def run_mapping(pmeta, template_html, llm, out_dir):
    print('[MappingAgent]')
    template = parse_template(template_html); print(f"  {len(template['slots'])} slots on {template['canvas_width']}x{template['canvas_height']}")
    mappings, tok = map_content(pmeta, template, llm)
    html, stats = fill_template(template, mappings, pmeta)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    assets = Path(template_html).parent / 'assets'
    if assets.is_dir(): shutil.copytree(assets, out / 'assets', dirs_exist_ok=True)      # decorative assets the mapper kept
    (out / 'content_mapping.json').write_text(json.dumps({'slot_mappings': mappings, 'stats': stats, 'tokens': tok}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'poster_filled.html').write_text(html, encoding='utf-8')
    return html, template, {'mappings': mappings, 'stats': stats, 'tokens': tok}
