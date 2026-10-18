"""ParserAgent: a paper (PDF or LaTeX zip) -> P_meta = {P_d, P_t, P_m, P_s, P_parse}.

P_parse = {meta, abstract, sections, figures, tables, formulas} plus two registries of data URIs (figures; rendered
equations/tables). P_d/P_t/P_m are taxonomy labels drawn from the RAP-25K vocabulary and P_s is the introduction
summary (motivation, problem formulation, key insight, claimed contributions) used for narrative matching.
"""
import base64
import difflib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined
from PIL import Image

from ..llm import extract_json

PROMPTS = Path(__file__).resolve().parent.parent / 'prompts'
MIN_FIGURE_AREA, MIN_FIGURE_DIM = 10_000, 80          # drop icons / logos among extracted pictures


def render_prompt(name, **kw):
    cfg = yaml.safe_load(open(PROMPTS / f'{name}.yaml', encoding='utf-8'))
    return cfg['system_prompt'], Environment(undefined=StrictUndefined).from_string(cfg['template']).render(**kw)


def image_data_uri(path, max_width=1200, quality=85):
    with Image.open(path) as im:
        if im.mode in ('RGBA', 'LA', 'P'): im, fmt, mime = im.convert('RGBA'), 'PNG', 'image/png'
        else: im, fmt, mime = im.convert('RGB'), 'JPEG', 'image/jpeg'
        if im.width > max_width: im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format=fmt, quality=quality)
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------------------------------- PDF input
def parse_pdf(pdf_path, out_dir, llm, formula_detector=None, figure_max_width=1200):
    """docling: markdown + figure/table images; optional formula crops; LLM condenses the text into poster sections."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem, TableItem

    out_dir = Path(out_dir); img_dir = out_dir / 'extracted_images'; img_dir.mkdir(parents=True, exist_ok=True)
    opts = PdfPipelineOptions(); opts.images_scale = 5.0; opts.generate_page_images = False; opts.generate_picture_images = True
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}).convert(str(pdf_path))
    markdown = re.sub(r'<!--[\s\S]*?-->', '', conv.document.export_to_markdown())
    figures, tables, nf, nt = [], [], 0, 0
    for element, _ in conv.document.iterate_items():
        if isinstance(element, TableItem):
            try:
                img = element.get_image(conv.document); nt += 1; p = img_dir / f'table_{nt}.png'; img.save(p, 'PNG')
                tables.append({'table_id': nt, 'caption': element.caption_text(conv.document), 'image_path': str(p), 'label': f'tab:{nt}'})
            except Exception as e:                                                                   # noqa: BLE001
                print(f'  table skipped: {e}')
        elif isinstance(element, PictureItem):
            try:
                img = element.get_image(conv.document)
                if img is None or img.width * img.height < MIN_FIGURE_AREA or min(img.size) < MIN_FIGURE_DIM: continue
                if not 0.15 <= img.width / max(1, img.height) <= 6.0: continue
                nf += 1; p = img_dir / f'figure_{nf}.png'; img.save(p, 'PNG')
                figures.append({'figure_id': nf, 'caption': element.caption_text(conv.document), 'graphics_paths': [f'figure_{nf}'],
                                'label': f'fig:{nf}', 'image_path': str(p), 'width': img.width, 'height': img.height})
            except Exception as e:                                                                   # noqa: BLE001
                print(f'  figure skipped: {e}')
    print(f'  docling: {len(markdown)} chars, {len(figures)} figures, {len(tables)} tables')
    figure_registry = {f'figure_{f["figure_id"]}': {'png_path': f['image_path'], 'width': f['width'], 'height': f['height'],
                                                   'base64_uri': image_data_uri(f['image_path'], figure_max_width)} for f in figures}
    rendered = {f'table_{t["table_id"]}': image_data_uri(t['image_path'], 1600) for t in tables}
    formulas = formula_detector(pdf_path, out_dir) if formula_detector else {}
    for k, p in formulas.items(): rendered[k] = image_data_uri(p, 1200)
    if formulas: print(f'  formulas: {len(formulas)}')
    system, prompt = render_prompt('pdf_parser_agent', markdown_content=markdown[:80_000])
    r = llm.text(prompt, system=system, temperature=0.3, max_tokens=16384); summary = extract_json(r['content'])
    for s in summary.get('sections', []): s.setdefault('subsections', [])
    parsed = {'meta': summary.get('meta', {'poster_title': '', 'authors': '', 'affiliations': ''}), 'abstract': summary.get('abstract', ''),
              'sections': summary.get('sections', []), 'figures': figures, 'tables': tables, 'formulas': sorted(formulas),
              '_raw_sections': _markdown_sections(markdown), '_tokens': {'input': r['input_tokens'], 'output': r['output_tokens']}}
    return parsed, figure_registry, rendered


def _markdown_sections(markdown, max_chars=4000):
    """Raw section text (from markdown headings) for the mapper's 'detailed content' block and the intro summary."""
    out, cur = [], None
    for line in markdown.splitlines():
        m = re.match(r'^(#{1,3})\s+(.*)', line)
        if m:
            cur = {'title': m.group(2).strip(), 'content': '', 'subsections': []}; out.append(cur)
        elif cur is not None: cur['content'] += line + '\n'
    for s in out: s['content'] = s['content'].strip()[:max_chars]
    return [s for s in out if s['content']]


# -------------------------------------------------------------------------------------------------- LaTeX input
def parse_latex(zip_path, out_dir, llm, figure_max_width=1200):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from latex.latex_parser import parse_latex_project
    from latex.figure_utils import build_figure_registry
    from latex.latex_renderer import render_equations_and_tables
    src = Path(out_dir) / 'latex_src'; src.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z: z.extractall(src)
    raw = parse_latex_project(str(src))
    figure_registry = build_figure_registry(str(src), max_width=figure_max_width)
    rendered = render_equations_and_tables(dict(raw, _raw_sections=raw['sections']), latex_dir=str(src))
    system, prompt = render_prompt('latex_parser_agent', meta=raw['meta'], abstract=raw['abstract'], sections=raw['sections'])
    r = llm.text(prompt, system=system, temperature=0.3, max_tokens=16384); summary = extract_json(r['content'])
    for s in summary.get('sections', []): s.setdefault('subsections', [])
    parsed = {'meta': summary.get('meta', raw['meta']), 'abstract': raw['abstract'], 'sections': summary.get('sections', []),
              'figures': raw['figures'], 'tables': raw['tables'], 'formulas': [k for k in rendered if k.startswith('equation_')],
              '_raw_sections': raw['sections'], '_tokens': {'input': r['input_tokens'], 'output': r['output_tokens']}}
    return parsed, figure_registry, rendered


# ------------------------------------------------------------------------------------- taxonomy + intro summary
def classify_and_summarize(parsed, taxonomy, llm):
    """P_d, P_t, P_m from the RAP-25K vocabulary and the narrative summary P_s."""
    raw = parsed.get('_raw_sections') or parsed['sections']
    intro = next((s['content'] for s in raw if re.search(r'introduction|motivation', s['title'], re.I)), raw[0]['content'] if raw else '')
    system, prompt = render_prompt('taxonomy_agent', title=parsed['meta'].get('poster_title', ''), abstract=parsed.get('abstract', '')[:2500],
                                   section_titles='; '.join(s['title'] for s in raw[:20]), introduction=intro[:6000],
                                   domains=taxonomy['domain'], tasks=taxonomy['task'], methods=taxonomy['method'])
    r = llm.text(prompt, system=system, temperature=0.1, max_tokens=2048); ans = extract_json(r['content'])
    labels = {}
    for k in ('domain', 'task', 'method'):
        v = str(ans.get(k, '')).strip(); vocab = taxonomy[k]
        labels[k] = v if v in vocab else (difflib.get_close_matches(v, vocab, n=1, cutoff=0.0) or [''])[0]
    return labels, str(ans.get('intro_summary', '')).strip(), {'input': r['input_tokens'], 'output': r['output_tokens']}


def parse_paper(path, out_dir, llm, taxonomy, formula_detector=None):
    path = Path(path); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[ParserAgent] {path.name}')
    if path.suffix.lower() == '.zip': parsed, figure_registry, rendered = parse_latex(path, out_dir, llm)
    else: parsed, figure_registry, rendered = parse_pdf(path, out_dir, llm, formula_detector)
    labels, intro_summary, tok = classify_and_summarize(parsed, taxonomy, llm)
    print(f'  labels: {labels}')
    pmeta = {'labels': labels, 'intro_summary': intro_summary, 'parsed': parsed, 'figure_registry': figure_registry, 'rendered_registry': rendered,
             'tokens': {'parse': parsed.pop('_tokens', {}), 'taxonomy': tok}}
    (out_dir / 'parsed_content.json').write_text(json.dumps({k: v for k, v in parsed.items() if k != '_raw_sections'}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'pmeta.json').write_text(json.dumps({'labels': labels, 'intro_summary': intro_summary, 'figures': list(figure_registry), 'rendered': list(rendered)}, indent=2, ensure_ascii=False), encoding='utf-8')
    return pmeta


def layoutlmv3_formula_detector(pdf_path, out_dir):
    """Optional formula crops via the repo's inference/ layout detector (needs its layout environment, see inference/README)."""
    repo = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(repo / 'inference'))
    from layout_runtime import extract
    dest = Path(out_dir) / 'layout'; extract(pdf_path, dest)
    assets = json.loads((dest / 'assets.json').read_text())['assets']
    return {a['id']: str(dest / 'assets' / f"{a['id']}.png") for a in assets if a['id'].startswith('formula_')}
