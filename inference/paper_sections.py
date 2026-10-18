#!/usr/bin/env python3
"""
Section-aware paper text for the poster SFT input (v2 of paper_side.load_paper_text).

Why this exists.  load_paper_text reads the LayoutLMv3 text boxes in page order
and stops at 12,000 characters.  Measured on the built dataset that cut lands
inside Related Work / Preliminaries for almost every paper: only 7.5% of inputs
contained any Experiments/Results section and 0.6% a Conclusion, while the
posters the model is trained to emit are mostly Method / Results / Conclusion.
Only 13.6% of poster text blocks shared a 7-gram with the input.

What it does instead -- the way a person reads a paper before making a poster:

  1. read every text box (same LayoutLMv3 classes, same reading order) up to
     References / Acknowledgements / Appendix, never a fixed character cut;
  2. split the stream into sections at the `title` boxes, work out each
     heading's level (numbering) and *role* (introduction, background, method,
     experiments, conclusion, ...);
  3. spend a per-role character budget: Introduction, Method, Experiments and
     Conclusion get most of it, Related Work gets none, and inside a section
     every subsection keeps its opening paragraphs rather than the first
     subsection eating the whole budget;
  4. tell the model which section each figure/table lives in, so a figure in
     the Method section reads as the overview figure and one in Experiments as
     a result.

The output is still plain text with `## ` headings, now `## [role] heading`,
so the prompt format changes only inside the <paper> and <assets> blocks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from paper_side import TEXT_CLASSES, reading_order

# PyMuPDF returns the PDF's own characters: typographic ligatures (fi fl) and the
# line-break hyphens of justified text ("calibra- tion") come through verbatim.
# Measured on 800 built inputs: 6.5 hyphenation artefacts and ~3 ligatures per
# 1000 words -- text the model would otherwise learn to copy into posters.
_LIGATURES = {'\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi', '\ufb04': 'ffl', '\ufb05': 'st', '\ufb06': 'st'}
_PUA = re.compile('[\ue000-\uf8ff]')
_HYPHEN_BREAK = re.compile(r'\b([A-Za-z]+)- ([a-z]+)\b')


def clean_text(t: str) -> str:
    for k, v in _LIGATURES.items():               # not NFKC: that would also flatten x^2 and R^d
        t = t.replace(k, v)
    t = _PUA.sub('', t)                           # private-use glyphs (broken symbol fonts)
    return re.sub(r'\s+', ' ', t).strip()


def dehyphenate(texts: List[str]) -> List[str]:
    """Resolve line-break hyphens with the paper's own vocabulary.

    "calibra- tion" -> "calibration" (the joined word occurs elsewhere, or neither
    form does: a syllable break); "self- supervised" -> "self-supervised" (the
    hyphenated form occurs elsewhere in the paper: a real compound).
    """
    joined_vocab, hyphen_vocab = set(), set()
    for t in texts:
        for w in re.findall(r'[A-Za-z]+(?:-[A-Za-z]+)+', t):
            hyphen_vocab.add(w.lower())
        for w in re.findall(r'\b[A-Za-z]{4,}\b', t):
            joined_vocab.add(w.lower())

    def fix(m):
        a, b = m.group(1), m.group(2)
        if (a + b).lower() in joined_vocab:
            return a + b
        if (a + '-' + b).lower() in hyphen_vocab:
            return a + '-' + b
        return a + b
    return [_HYPHEN_BREAK.sub(fix, t) for t in texts]

# --------------------------------------------------------------------------- #
# budgets (characters).  ~3.4 chars/token for this kind of English prose.
# --------------------------------------------------------------------------- #
#
# Measured with the Qwen3.8 tokenizer on the test split: this prose runs ~2.5
# chars/token (math, citations), so 22k chars ≈ 9k text tokens; with ~2.2k
# visual tokens and a ~3.8k-token target the whole sample sits ~11.5k tokens
# median, p90 ≈ 13.5k, inside a 16384 window with a small margin.
ROLE_BUDGET = {
    'abstract': 0,          # already carried in the <paper> header
    'introduction': 4000,
    'related': 0,           # posters never carry it
    'background': 1500,
    'method': 8000,
    'experiments': 7000,
    'conclusion': 2000,
    'other': 1200,
}
TOTAL_BUDGET = 22000
SUBSECTION_FLOOR = 500      # every subsection keeps at least its opening
SECTION_FLOOR = 800
STOP_ROLES = {'references', 'acknowledgements', 'appendix'}

_RULES: List[Tuple[str, str]] = [
    ('references', r'\b(references?|bibliography)\b'),
    ('acknowledgements', r'\backnowledg'),
    ('appendix', r'\b(appendix|appendices|supplementa)'),
    ('conclusion', r'\b(conclusions?|concluding|discussions?|limitations?|future work|'
                   r'summary|broader impacts?|societal impact|ethic)'),
    ('experiments', r'\b(experiments?|experimental|results?|evaluations?|empirical|'
                    r'benchmarks?|ablations?|case stud|applications?|findings)'),
    ('related', r'\b(related|prior|previous|existing) work|literature'),
    ('introduction', r'\bintroduction\b|\bmotivation\b'),
    ('abstract', r'\babstract\b'),
    ('background', r'\b(preliminar|background|notation|problem (setup|formulation|'
                   r'statement|definition|setting)|setup|setting|definitions?|formulation)'),
]
_NUM = re.compile(r'^\s*(\d+(?:\.\d+)*)[.\s]')
_ROMAN = re.compile(r'^\s*(?:[IVX]+)[.\s]+[A-Z]')
_LETTER = re.compile(r'^\s*[A-Z][.)]\s+[A-Z]')


def _role_of(text: str) -> Optional[str]:
    t = text.lower()
    for role, pat in _RULES:
        if re.search(pat, t):
            return role
    return None


def _level_of(text: str, role: Optional[str], have_l1: bool) -> int:
    m = _NUM.match(text)
    if m:
        return min(m.group(1).count('.') + 1, 3)
    if _ROMAN.match(text):
        return 1
    if _LETTER.match(text):
        return 2
    if role in ('abstract', 'introduction', 'related', 'conclusion') or role in STOP_ROLES:
        return 1
    return 2 if have_l1 else 1


def _looks_like_heading(text: str) -> bool:
    if len(text) > 140 or len(text) < 3:
        return False
    if re.match(r'^(theorem|lemma|corollary|proposition|definition|proof|remark|'
                r'assumption|example|algorithm|figure|table)\b', text.lower()):
        return False
    return True


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def read_items(pid: str, root: str, pdf_path: str, max_pages: int = 12) -> Tuple[List[Dict], Dict]:
    """Text boxes in reading order: [{'kind','text','page','bbox','key'}], plus page info.

    `key` is a sortable reading-order position (page, column, y) shared with
    assets so a figure can be attributed to the section it sits in.
    """
    d = Path(root) / pid
    layout_p = d / 'layout.json'
    if not layout_p.exists() or not pdf_path or not Path(pdf_path).exists():
        return [], {}
    layout = json.loads(layout_p.read_text(encoding='utf-8'))
    meta = json.loads((d / 'assets.json').read_text(encoding='utf-8'))
    scale = 72.0 / meta.get('dpi', 200)

    import fitz
    doc = fitz.open(pdf_path)
    items: List[Dict] = []
    pages: Dict[int, Dict] = {}
    try:
        for page_no in range(1, min(max_pages, len(doc)) + 1):
            boxes = [b for b in layout if b.get('page') == page_no
                     and b.get('class_name') in TEXT_CLASSES]
            page = doc[page_no - 1]
            pw = page.rect.width / scale
            ph = page.rect.height / scale
            pages[page_no] = {'w': pw, 'h': ph}
            if not boxes:
                continue
            ordered = reading_order(boxes, pw)
            body_w = max(b['bbox'][2] - b['bbox'][0] for b in boxes)
            for b in ordered:
                x1, y1, x2, y2 = (v * scale for v in b['bbox'])
                t = page.get_text('text', clip=fitz.Rect(x1 - 1, y1 - 1, x2 + 1, y2 + 1))
                t = clean_text(t)
                if len(t) < 3:
                    continue
                items.append({'kind': 'title' if TEXT_CLASSES[b['class_name']] == 'title' else 'text',
                              'text': t, 'page': page_no, 'bbox': b['bbox'],
                              'key': order_key(b['bbox'], page_no, pw, body_w)})
    finally:
        doc.close()
    fixed = dehyphenate([it['text'] for it in items])
    for it, t in zip(items, fixed):
        it['text'] = t
    return items, pages


def order_key(bbox, page_no: int, page_w: float, body_w: float) -> Tuple[int, int, float]:
    x1, y1, x2, _ = bbox
    spanning = (x2 - x1) > 0.75 * body_w
    col = 0 if spanning or (x1 + x2) / 2 < page_w / 2 else 1
    return (page_no, col, float(y1))


def split_sections(items: List[Dict]) -> List[Dict]:
    """-> [{'heading','level','role','page','key','groups':[{'heading','level','paras':[str]}]}]

    Level-1 sections carry a role; deeper headings become groups inside the
    enclosing level-1 section.  Text before the first real heading (paper
    title, authors, affiliations, and the abstract when it has no heading box)
    is kept in a 'front' section that the budget step drops.
    """
    sections: List[Dict] = []
    cur: Optional[Dict] = None
    seen_intro = seen_exp = False
    front = {'heading': '', 'level': 1, 'role': 'front', 'page': 0, 'key': (0, 0, 0.0),
             'groups': [{'heading': '', 'level': 1, 'paras': []}]}

    def new_section(heading, level, role, it):
        return {'heading': heading, 'level': level, 'role': role, 'page': it['page'],
                'key': it['key'], 'groups': [{'heading': '', 'level': level, 'paras': []}]}

    for it in items:
        if it['kind'] == 'title' and _looks_like_heading(it['text']):
            role = _role_of(it['text'])
            level = _level_of(it['text'], role, cur is not None)
            if level == 1:
                if role is None:
                    # positional default: between intro and experiments = method,
                    # after experiments = still experiments (Analysis, Discussion...)
                    role = 'experiments' if seen_exp else ('method' if seen_intro else 'other')
                if role in STOP_ROLES:
                    break
                if not seen_intro and role not in ('abstract', 'introduction') and not _NUM.match(it['text']):
                    # paper title, author names, affiliations: front matter, not a section
                    role = 'front'
                seen_intro |= role in ('introduction', 'abstract')
                seen_exp |= role == 'experiments'
                cur = new_section(it['text'], 1, role, it)
                sections.append(cur)
            else:
                if cur is None:           # sub-heading before any section: treat as front text
                    front['groups'][0]['paras'].append(it['text'])
                    continue
                cur['groups'].append({'heading': it['text'], 'level': level, 'paras': []})
            continue
        target = cur if cur is not None else front
        target['groups'][-1]['paras'].append(it['text'])
    return [front] + sections


# --------------------------------------------------------------------------- #
# budgeting
# --------------------------------------------------------------------------- #

def _cut(text: str, n: int) -> str:
    """Cut at a sentence boundary at or before n characters."""
    if len(text) <= n:
        return text
    head = text[:n]
    m = max(head.rfind('. '), head.rfind('? '), head.rfind('! '))
    if m > n * 0.5:
        return head[:m + 1]
    m = head.rfind(' ')
    return (head[:m] if m > 0 else head) + ' …'


def _take_paras(paras: List[str], budget: int) -> Tuple[str, bool]:
    out, used, cut = [], 0, False
    for p in paras:
        if used + len(p) + 2 <= budget:
            out.append(p)
            used += len(p) + 2
            continue
        room = budget - used
        if room > 200:
            out.append(_cut(p, room))
        cut = True
        break
    return '\n\n'.join(out), cut


def _spend(groups: List[Dict], budget: int) -> str:
    """Distribute a section's budget over its subsection groups, floor first."""
    lens = [sum(len(p) + 2 for p in g['paras']) for g in groups]
    total = sum(lens)
    if total == 0:
        return ''
    if total <= budget:
        alloc = lens
    else:
        n = len(groups)
        floor = min(SUBSECTION_FLOOR, budget // max(n, 1))
        alloc = [min(l, floor) for l in lens]
        rest = budget - sum(alloc)
        extra_need = [l - a for l, a in zip(lens, alloc)]
        need_total = sum(extra_need)
        if rest > 0 and need_total > 0:
            alloc = [a + int(rest * e / need_total) for a, e in zip(alloc, extra_need)]
    parts = []
    for g, b in zip(groups, alloc):
        mark = '#' * max(g['level'] + 1, 3)          # section = ##, subsections = ###, ####
        if not g['paras']:
            if g['heading']:
                parts.append(mark + ' ' + g['heading'])
            continue
        body, cut = _take_paras(g['paras'], b)
        if not body:
            continue
        head = (mark + ' ' + g['heading'] + '\n\n') if g['heading'] else ''
        parts.append(head + body + (' […]' if cut else ''))
    return '\n\n'.join(parts)


def budget_body(sections: List[Dict], total: int = TOTAL_BUDGET,
                role_budget: Dict[str, int] = ROLE_BUDGET,
                abstract_missing: bool = False) -> Tuple[str, Dict]:
    """Render the sections into the body text under the per-role budgets."""
    if total < TOTAL_BUDGET:
        # long poster / many figures: shrink every role's share proportionally
        # so the section mix stays the same and only the depth per section drops
        f = total / TOTAL_BUDGET
        role_budget = {k: int(v * f) for k, v in role_budget.items()}
    keep = [s for s in sections if s['role'] not in ('front', 'related', 'abstract')]
    if abstract_missing:
        for s in sections:
            if s['role'] in ('abstract', 'front'):
                keep.insert(0, dict(s, role='abstract'))
                break
    lens = {id(s): sum(len(p) + 2 for g in s['groups'] for p in g['paras']) for s in keep}
    by_role: Dict[str, List[Dict]] = {}
    for s in keep:
        by_role.setdefault(s['role'], []).append(s)

    alloc: Dict[int, int] = {}
    for role, secs in by_role.items():
        b = role_budget.get(role, role_budget['other'])
        if role == 'abstract':
            b = 1800
        need = [lens[id(s)] for s in secs]
        if sum(need) <= b:
            for s, n in zip(secs, need):
                alloc[id(s)] = n
        else:
            floor = min(SECTION_FLOOR, b // max(len(secs), 1))
            a = [min(n, floor) for n in need]
            rest = b - sum(a)
            extra = [n - x for n, x in zip(need, a)]
            et = sum(extra)
            for i, s in enumerate(secs):
                alloc[id(s)] = a[i] + (int(rest * extra[i] / et) if et > 0 and rest > 0 else 0)
    # leftover (no conclusion, short intro...) goes to experiments, then method
    used = sum(alloc.values())
    for role in ('experiments', 'method', 'introduction'):
        rest = total - used
        if rest <= 0:
            break
        for s in by_role.get(role, []):
            room = lens[id(s)] - alloc[id(s)]
            if room > 0:
                add = min(room, rest)
                alloc[id(s)] += add
                used += add
                rest -= add
    # hard cap
    if used > total:
        scale = total / used
        alloc = {k: int(v * scale) for k, v in alloc.items()}

    parts, stats = [], {}
    for s in keep:
        text = _spend(s['groups'], alloc[id(s)])
        if not text.strip():
            continue
        parts.append(f"## [{s['role']}] {s['heading']}".rstrip() + '\n\n' + text)
        stats[s['role']] = stats.get(s['role'], 0) + len(text)
    body = '\n\n'.join(parts)
    stats['total'] = len(body)
    stats['roles'] = sorted(by_role)
    return body, stats


def load_paper_body(pid: str, root: str, pdf_path: str, abstract_missing: bool = False,
                    max_pages: int = 12, total: int = TOTAL_BUDGET) -> Tuple[str, Dict, List[Dict]]:
    items, _ = read_items(pid, root, pdf_path, max_pages)
    if not items:
        return '', {'total': 0, 'roles': []}, []
    sections = split_sections(items)
    body, stats = budget_body(sections, total=min(total, TOTAL_BUDGET), abstract_missing=abstract_missing)
    stats['budget'] = min(total, TOTAL_BUDGET)
    return body, stats, sections


# --------------------------------------------------------------------------- #
# assets: section attribution + caption fallback
# --------------------------------------------------------------------------- #

def section_of_asset(sections: List[Dict], page: int, bbox, page_w: float, body_w: float) -> Optional[Dict]:
    key = order_key(bbox, page, page_w, body_w)
    best = None
    for s in sections:
        if s['role'] == 'front':
            continue
        if s['key'] <= key:
            best = s
    return best


_CAP_LINE = re.compile(r'^\s*(Figure|Fig\.|Table)\s*(\d+|[IVX]+)\s*[:.]', re.I)


def section_by_reference(sections: List[Dict], caption: str, kind: str) -> Optional[Dict]:
    """The section whose text first cites this figure/table ("Figure 3", "Tab. 1").

    Figures sit where the page layout allows (often at the top of the next
    column, under the next heading); they are *discussed* in the section that
    cites them, and that is the section a poster would place them in.
    """
    m = _CAP_LINE.match(caption or '')
    if not m or kind not in ('figure', 'table'):
        return None
    num = m.group(2)
    if kind == 'table':
        pat = re.compile(r'\b(Table|Tab\.)\s*' + re.escape(num) + r'(?!\d)')
    else:
        pat = re.compile(r'\b(Figure|Fig\.|Figs?\.)\s*' + re.escape(num) + r'(?!\d)')
    for s in sections:
        if s['role'] in ('front', 'related'):
            continue
        for g in s['groups']:
            for p in g['paras']:
                if pat.search(p):
                    return s
    return None


def caption_fallback(doc, page_no: int, bbox, kind: str, scale: float, max_chars: int = 300) -> str:
    """When no LayoutLMv3 caption box matched: the nearest 'Figure N:'/'Table N:'
    text block on the page, below the asset (tables: above or below)."""
    if page_no > len(doc):
        return ''
    page = doc[page_no - 1]
    ph = page.rect.height / scale
    ax1, ay1, ax2, ay2 = bbox
    want = 'table' if kind == 'table' else 'figure'
    best, best_gap = '', 1e9
    for blk in page.get_text('dict').get('blocks', []):
        lines = blk.get('lines') or []
        if not lines:
            continue
        text = ' '.join(s['text'] for l in lines for s in l['spans'])
        text = re.sub(r'\s+', ' ', text).strip()
        m = _CAP_LINE.match(text)
        if not m:
            continue
        if (m.group(1).lower().startswith('tab')) != (want == 'table'):
            continue
        bx1, by1, bx2, by2 = (v / scale for v in blk['bbox'])
        lo, hi = max(ax1, bx1), min(ax2, bx2)
        if hi <= lo:
            continue
        gap = by1 - ay2 if by1 >= ay2 else ay1 - by2
        if gap < -5 or gap > 0.2 * ph:
            continue
        if gap < best_gap:
            best, best_gap = text, gap
    if len(best) > max_chars:
        best = best[:max_chars].rsplit(' ', 1)[0] + '…'
    return best
