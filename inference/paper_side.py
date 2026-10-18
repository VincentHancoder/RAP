#!/usr/bin/env python3
"""
The paper side of a training sample: which assets exist, and what each is called.

Two things here are not what the earlier docling-based version did.

**Assets come from the LayoutLMv3 registry.** The poster side was derendered
with LayoutLMv3, so the paper side has to use it too or the crops never line up
-- that asymmetry is what made the first matching attempt useless. The registry
also carries page and bbox, which the caption logic needs.

**Captions are read out of the PDF at the caption box's own coordinates.**
Pairing the Nth crop with the Nth caption in a parsed-content dump is what put
"Figure 3"'s text on figure_4 earlier: any dropped detection on either side
shifts every later pairing silently. LayoutLMv3 already emits figure_caption /
table_caption boxes, so instead of counting, this finds the caption box nearest
*below* each asset on the same page and pulls the text inside that rectangle
straight from the PDF. A caption either has a box under its figure or the asset
goes uncaptioned; it can no longer be given someone else's.

    from paper_side import load_paper_assets
    assets = load_paper_assets(pid, paper_assets_root, pdf_path)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

CAPTION_FOR = {'figure': 'figure_caption', 'table': 'table_caption',
               'isolate_formula': 'formula_caption'}
KIND_NAME = {'figure': 'figure', 'table': 'table', 'isolate_formula': 'formula'}
# A caption sits directly under (tables: often over) its asset. Anything further
# than this fraction of the page height away belongs to something else.
MAX_CAPTION_GAP = 0.12
MAX_CAPTION_CHARS = 300


def _overlap(a, b) -> float:
    """Horizontal overlap of two boxes as a fraction of the narrower one."""
    lo, hi = max(a[0], b[0]), min(a[2], b[2])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(min(a[2] - a[0], b[2] - b[0]), 1e-6)


def _clean(t: str) -> str:
    t = re.sub(r'\s+', ' ', t).strip()
    if len(t) > MAX_CAPTION_CHARS:
        t = t[:MAX_CAPTION_CHARS].rsplit(' ', 1)[0] + '…'
    return t


def load_paper_assets(pid: str, root: str, pdf_path: Optional[str] = None,
                      main_body_pages: int = 12) -> Optional[Dict]:
    """-> {'assets': [...], 'counts': {...}, 'n_pages': int} or None."""
    d = Path(root) / pid
    meta_p = d / 'assets.json'
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding='utf-8'))
    except Exception:                                                # noqa: BLE001
        return None

    assets = []
    for a in meta.get('assets', []):
        f = d / 'assets' / f"{a['id']}.png"
        if not f.exists():
            continue
        assets.append({'id': a['id'], 'class': a['class'], 'kind': KIND_NAME[a['class']],
                       'page': a['page'], 'bbox': a['bbox'], 'aspect': a.get('aspect'),
                       'area': (a['bbox'][2] - a['bbox'][0]) * (a['bbox'][3] - a['bbox'][1]),
                       'file': str(f), 'caption': '',
                       'main_body': a['page'] <= main_body_pages})

    if pdf_path:
        try:
            _attach_captions(assets, d, pdf_path, meta.get('dpi', 200))
        except Exception:                                            # noqa: BLE001
            pass                       # captions are a nicety, assets are not

    return {'paper_id': pid, 'assets': assets, 'counts': meta.get('counts', {}),
            'n_pages': meta.get('n_pages', 0)}


def _attach_captions(assets: List[Dict], paper_dir: Path, pdf_path: str,
                     dpi: int) -> None:
    layout_p = paper_dir / 'layout.json'
    if not layout_p.exists():
        return
    layout = json.loads(layout_p.read_text(encoding='utf-8'))
    caps: Dict[int, List[Dict]] = {}
    for d in layout:
        if d.get('class_name') in CAPTION_FOR.values():
            caps.setdefault(d['page'], []).append(d)
    if not caps:
        return

    import fitz
    doc = fitz.open(pdf_path)
    scale = 72.0 / dpi              # rendered pixels back to PDF points
    try:
        for a in assets:
            want = CAPTION_FOR.get(a['class'])
            page_caps = [c for c in caps.get(a['page'], [])
                         if c.get('class_name') == want]
            if not page_caps or a['page'] > len(doc):
                continue
            page = doc[a['page'] - 1]
            ph = page.rect.height / scale       # page height in rendered pixels
            ax1, ay1, ax2, ay2 = a['bbox']
            best, best_gap = None, 1e9
            for c in page_caps:
                cx1, cy1, cx2, cy2 = c['bbox']
                if _overlap((ax1, ay1, ax2, ay2), (cx1, cy1, cx2, cy2)) < 0.3:
                    continue
                # below the asset, or above it (the usual place for a table)
                gap = cy1 - ay2 if cy1 >= ay2 else ay1 - cy2
                if gap < -5 or gap > MAX_CAPTION_GAP * ph:
                    continue
                if gap < best_gap:
                    best, best_gap = c, gap
            if best is None:
                continue
            x1, y1, x2, y2 = (v * scale for v in best['bbox'])
            txt = page.get_text('text', clip=fitz.Rect(x1 - 2, y1 - 2, x2 + 2, y2 + 2))
            a['caption'] = _clean(txt)
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# paper text
# --------------------------------------------------------------------------- #

# LayoutLMv3's own vocabulary. `abandon` is what it calls running heads, page
# numbers and footers -- exactly the text a poster never carries, and exactly
# what a raw page-level text dump would splice into the middle of a paragraph.
TEXT_CLASSES = {'title': 'title', 'plain text': 'text', 'plain_text': 'text'}
DROP_CLASSES = {'abandon', 'figure', 'table', 'isolate_formula',
                'figure_caption', 'table_caption', 'formula_caption',
                'table_footnote'}
MAX_BODY_CHARS = 12000


SPAN_FRAC = 0.75          # of the page's own text width, not the paper size


def is_two_column(boxes: List[Dict], mid: float) -> bool:
    """Does this page actually have two columns?

    Deciding by "is the centre left or right of the page middle" invents a
    gutter on single-column pages, and a wide-margin single-column layout is
    common enough (ICLR's template among them) that the invented gutter
    reordered abstracts after introductions. A real two-column page has boxes
    that finish before the middle *and* boxes that start after it.
    """
    return (any(b['bbox'][2] < mid for b in boxes)
            and any(b['bbox'][0] > mid for b in boxes))


def reading_order(boxes: List[Dict], page_width: float) -> List[Dict]:
    """Order text boxes the way the page is read.

    Two corrections over sorting by (column, y):

    Spanning is measured against the widest box on the page, not the paper
    size. A single-column ICLR page is 1700 px wide but its text column is
    1115 px, so a page-relative threshold classified ordinary body paragraphs
    as spanning and full-width titles as not.

    Column splitting only happens on pages that have two columns. Otherwise
    everything is read top to bottom, which is what a single-column page wants
    and what the gutter heuristic actively broke.

    On a genuine two-column page a spanning box acts as a barrier: pending
    column boxes are emitted left-then-right before it, so full-width headers
    stay in place and the columns beneath them stay in order.
    """
    if not boxes:
        return []
    body_w = max(b['bbox'][2] - b['bbox'][0] for b in boxes)
    span_w = SPAN_FRAC * body_w
    ordered = sorted(boxes, key=lambda b: b['bbox'][1])

    mid = page_width / 2
    narrow = [b for b in ordered if (b['bbox'][2] - b['bbox'][0]) <= span_w]
    if not is_two_column(narrow, mid):
        return ordered

    out: List[Dict] = []
    pending: List[Dict] = []

    def flush():
        if not pending:
            return
        left = [b for b in pending if (b['bbox'][0] + b['bbox'][2]) / 2 < mid]
        right = [b for b in pending if (b['bbox'][0] + b['bbox'][2]) / 2 >= mid]
        for group in (left, right):
            out.extend(sorted(group, key=lambda b: b['bbox'][1]))
        pending.clear()

    for b in ordered:
        if (b['bbox'][2] - b['bbox'][0]) > span_w:
            flush()
            out.append(b)
        else:
            pending.append(b)
    flush()
    return out


def load_paper_text(pid: str, root: str, pdf_path: str,
                    max_pages: int = 12, max_chars: int = MAX_BODY_CHARS) -> str:
    """Body text in reading order, taken from the detected text boxes.

    Reading text box by box rather than page by page buys three things the raw
    dump cannot: running heads and page numbers are dropped because LayoutLMv3
    labels them `abandon`; captions and equations are dropped because they are
    already carried as assets and would otherwise appear twice; and two-column
    papers come out in column order instead of interleaved across the gutter.

    Ordering is by column then vertical position, which is what a two-column
    layout needs and what a single-column layout is unaffected by.
    """
    d = Path(root) / pid
    layout_p = d / 'layout.json'
    if not layout_p.exists() or not pdf_path:
        return ''
    try:
        layout = json.loads(layout_p.read_text(encoding='utf-8'))
        meta = json.loads((d / 'assets.json').read_text(encoding='utf-8'))
    except Exception:                                                # noqa: BLE001
        return ''
    scale = 72.0 / meta.get('dpi', 200)

    try:
        import fitz
        doc = fitz.open(pdf_path)
    except Exception:                                                # noqa: BLE001
        return ''

    parts: List[str] = []
    try:
        for page_no in range(1, min(max_pages, len(doc)) + 1):
            boxes = [b for b in layout
                     if b.get('page') == page_no
                     and b.get('class_name') in TEXT_CLASSES
                     and b.get('class_name') not in DROP_CLASSES]
            if not boxes:
                continue
            page = doc[page_no - 1]
            pw = page.rect.width / scale
            for b in reading_order(boxes, pw):
                x1, y1, x2, y2 = (v * scale for v in b['bbox'])
                t = page.get_text('text', clip=fitz.Rect(x1 - 1, y1 - 1, x2 + 1, y2 + 1))
                t = re.sub(r'\s+', ' ', t).strip()
                if len(t) < 3:
                    continue
                parts.append(('## ' + t) if TEXT_CLASSES[b['class_name']] == 'title' else t)
                if sum(len(p) for p in parts) > max_chars:
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        doc.close()

    out = '\n\n'.join(parts)
    return out[:max_chars]


def select_for_prompt(assets: List[Dict], required: List[str],
                      max_total: int, per_kind: Dict[str, int]) -> List[Dict]:
    """Choose which assets the model is shown.

    Anything the target program references is mandatory: dropping it would make
    the sample ask for a figure that was never offered, teaching the model to
    hallucinate ids. Whatever budget is left goes to the largest remaining
    assets, since a poster overwhelmingly draws on a paper's headline figures.
    """
    by_id = {a['id']: a for a in assets}
    out = [by_id[i] for i in required if i in by_id]
    have = {a['id'] for a in out}
    kind_n: Dict[str, int] = {}
    for a in out:
        kind_n[a['kind']] = kind_n.get(a['kind'], 0) + 1

    for a in sorted(assets, key=lambda x: -x['area']):
        if len(out) >= max_total:
            break
        if a['id'] in have or not a['main_body']:
            continue
        k = a['kind']
        if kind_n.get(k, 0) >= per_kind.get(k, 99):
            continue
        out.append(a)
        have.add(a['id'])
        kind_n[k] = kind_n.get(k, 0) + 1

    # Stable, readable order for the prompt: figures, then tables, then formulas.
    order = {'figure': 0, 'table': 1, 'formula': 2}
    return sorted(out, key=lambda a: (order.get(a['kind'], 9), a['id']))
