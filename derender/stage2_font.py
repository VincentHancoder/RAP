"""Stage 2 — Font Size Estimation.

For every text-like layout box (title, plain text, captions, table footnote) the region is cropped and
read with PaddleOCR. OCR boxes are grouped into visual lines by vertical overlap (≥ 0.5 of the smaller
box) and the font size in CSS pixels is

    line_height_px = crop_h / n_lines
    font_size_px   = line_height_px * 0.8      # the glyph body is ~0.8 of the line box

The result is `layout_with_font.json`: the Stage 1 JSON with `n_lines`, `line_height_px`,
`font_size_px`, `texts`, `text` added to each text-like detection (image-like detections untouched).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

TEXT_LIKE_CLASSES = {"title", "plain text", "figure_caption", "table_caption", "table_footnote", "formula_caption"}
IMAGE_LIKE_CLASSES = {"figure", "abandon", "table", "isolate_formula"}


class OCR:
    """PaddleOCR wrapper returning [{'poly': [x1,y1,...,x4,y4], 'text': str, 'score': float}, ...]."""

    def __init__(self, lang: str = "ch", det_db_box_thresh: float = 0.3, det_model_dir=None, rec_model_dir=None):
        from paddleocr import PaddleOCR
        kwargs = dict(lang=lang)
        if det_model_dir: kwargs["det_model_dir"] = det_model_dir
        if rec_model_dir: kwargs["rec_model_dir"] = rec_model_dir
        self.legacy = True
        try:                                                   # PaddleOCR 2.x
            self.ocr = PaddleOCR(use_angle_cls=True, show_log=False, det_db_box_thresh=det_db_box_thresh, **kwargs)
        except (TypeError, ValueError):                        # PaddleOCR 3.x
            self.legacy = False
            self.ocr = PaddleOCR(use_textline_orientation=True, text_det_box_thresh=det_db_box_thresh, **kwargs)

    def predict(self, img: Image.Image) -> list[dict]:
        arr = np.array(img.convert("RGB"))
        out = []
        if self.legacy:
            res = self.ocr.ocr(arr, cls=True)
            if not res or res[0] is None:
                return out
            for pts, (text, score) in res[0]:
                out.append({"poly": [float(v) for p in pts for v in p], "text": text, "score": round(float(score), 2)})
        else:
            for r in self.ocr.predict(arr):
                for pts, text, score in zip(r.get("dt_polys", []), r.get("rec_texts", []), r.get("rec_scores", [])):
                    out.append({"poly": [float(v) for p in pts for v in p], "text": text, "score": round(float(score), 2)})
        return out


def crop_bbox(image: Image.Image, bbox) -> tuple[Image.Image, list[int]]:
    x1, y1, x2, y2 = (int(round(max(0.0, min(float(v), lim)))) for v, lim in zip(bbox, (image.width, image.height) * 2))
    left, right, top, bottom = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    right, bottom = max(right, left + 1), max(bottom, top + 1)
    return image.crop((left, top, right, bottom)), [left, top, right, bottom]


def count_lines(ocr_results: list[dict], min_overlap_ratio: float = 0.5) -> int:
    items = sorted(((min(r["poly"][1], r["poly"][3]), max(r["poly"][5], r["poly"][7])) for r in ocr_results), key=lambda t: t[0])
    lines: list[tuple[float, float]] = []
    for y0, y1 in items:
        for i, (ly0, ly1) in enumerate(lines):
            overlap = max(0.0, min(y1, ly1) - max(y0, ly0))
            if overlap / max(1e-3, min(y1 - y0, ly1 - ly0)) >= min_overlap_ratio:
                lines[i] = (min(ly0, y0), max(ly1, y1)); break
        else:
            lines.append((y0, y1))
    return len(lines)


def estimate_font(crop_h: int, n_lines: int, ratio: float = 0.8):
    if n_lines <= 0 or crop_h <= 0:
        return None, None
    line_h = crop_h / n_lines
    return round(line_h, 3), round(line_h * ratio, 3)


def run(image_path: Path, layout_json: Path, out_dir: Path, ocr: OCR, conf_thres: float = 0.25,
        ratio: float = 0.8, save_crops: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    pages = json.loads(layout_json.read_text(encoding="utf-8"))
    page = pages[0] if isinstance(pages, list) else pages
    (out_dir / "ocr_per_box").mkdir(exist_ok=True)
    if save_crops: (out_dir / "text_crops").mkdir(exist_ok=True)
    summary = []
    for idx, det in enumerate(page.get("detections", [])):
        bid = det.setdefault("id", f"blk-{idx + 1:04d}")
        if float(det.get("score", 0)) < conf_thres:
            det["ocr_skipped"] = "below_conf_thres"; continue
        if det.get("class_name") not in TEXT_LIKE_CLASSES:
            continue
        crop, clip = crop_bbox(image, det["bbox"])
        if save_crops: crop.save(out_dir / "text_crops" / f"{bid}.png")
        res = ocr.predict(crop)
        (out_dir / "ocr_per_box" / f"{bid}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        n = count_lines(res); line_h, font_px = estimate_font(crop.height, n, ratio)
        texts = [r["text"] for r in res]
        det.update(crop_bbox=clip, crop_w=crop.width, crop_h=crop.height, n_lines=n, line_height_px=line_h,
                   font_size_px=font_px, texts=texts, text="\n".join(t for t in texts if t))
        summary.append({"id": bid, "label": det["class_name"], "bbox": det["bbox"], "n_lines": n, "font_size_px": font_px,
                        "sample_text": det["text"][:80]})
    (out_dir / "layout_with_font.json").write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "font_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "conversion.json").write_text(json.dumps({
        "formula": ["line_height_px = crop_h / n_lines", "font_size_px = line_height_px * font_to_line_ratio"],
        "parameters": {"font_to_line_ratio": ratio, "line_count_min_overlap_ratio": 0.5, "conf_thres": conf_thres},
        "timestamp": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    return out_dir / "layout_with_font.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, nargs="+"); ap.add_argument("--layout_json", required=True, nargs="+")
    ap.add_argument("--output_dir", required=True, nargs="+")
    ap.add_argument("--lang", default="ch"); ap.add_argument("--det_db_box_thresh", type=float, default=0.3)
    ap.add_argument("--conf_thres", type=float, default=0.25); ap.add_argument("--font_to_line_ratio", type=float, default=0.8)
    ap.add_argument("--no_save_crops", action="store_true")
    a = ap.parse_args()
    ocr = OCR(a.lang, a.det_db_box_thresh)
    for img, lay, out in zip(a.input, a.layout_json, a.output_dir):
        p = run(Path(img), Path(lay), Path(out), ocr, a.conf_thres, a.font_to_line_ratio, not a.no_save_crops)
        print(f"FONT_DONE {Path(img).name}: {p}", flush=True)


if __name__ == "__main__":
    main()
