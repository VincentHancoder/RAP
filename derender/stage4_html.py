"""Stage 4 — HTML Reconstruction (+ render and compare).

An HTML scaffold is built programmatically from the Stage 1/2 layout: one absolutely positioned <div> per
detected block; image-like regions (figures, tables, formulae, logos) are cropped from the poster and
embedded as <img>, text-like blocks are empty slots annotated with their OCR-measured font size. The scaffold,
the poster image and the Stage 3 design caption go to a multimodal LLM, which fills every text slot by reading
the poster and synthesises the CSS. The result is rendered with headless Chromium and compared with the original.
Prompt and settings are the ones used to build RAP-25K. Outputs the RAP-25K template folder:
`generated.html`, `assets/`, `rendered.png`, `layout.json`, `design_free.json`, `run_config.json`.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .design_dimensions import DIMENSIONS
from .mllm import MLLM, MAX_TOKENS_HTML, TEMPERATURE

PROMPT_TEMPLATE = Path(__file__).resolve().parent / "prompts" / "html_reconstruction.txt"
SYSTEM_PROMPT = "You are a precise web reconstruction assistant. Return valid complete HTML only."
CONF_THRES = 0.25


def _get_dimensions():
    return DIMENSIONS


CLASS_ID_TO_NAME = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

TEXT_LIKE_CLASSES = {"title", "plain text", "figure_caption", "table_caption",
                     "table_footnote", "formula_caption"}
IMAGE_LIKE_CLASSES = {"figure", "abandon", "table", "isolate_formula"}
TABLE_LIKE_CLASSES = set()
FORMULA_LIKE_CLASSES = set()
SKIP_CLASSES: set[str] = set()

def save_crop_asset(image: Image.Image, bbox: list[int], path: Path) -> None:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(x2, image.width)
    y2 = min(y2, image.height)
    crop = image.crop((x1, y1, x2, y2))
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)

def classify_box(class_name: str) -> str:
    if class_name in TEXT_LIKE_CLASSES:
        return "text-like"
    if class_name in IMAGE_LIKE_CLASSES:
        return "image-like"
    if class_name in TABLE_LIKE_CLASSES:
        return "table-like"
    if class_name in FORMULA_LIKE_CLASSES:
        return "formula-like"
    if class_name in SKIP_CLASSES:
        return "skip"
    return "other"


def normalize_layout(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    conf_thres: float,
) -> list[dict[str, Any]]:
    """Convert run_layout.py detections into normalised boxes sorted top-to-bottom."""
    boxes: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        score = det.get("score", 0.0)
        if score < conf_thres:
            continue

        class_name = det.get("class_name", "unknown")
        kind = classify_box(class_name)
        if kind == "skip":
            continue

        raw = det["bbox"]
        x1 = int(round(max(0.0, min(float(raw[0]), image_width))))
        y1 = int(round(max(0.0, min(float(raw[1]), image_height))))
        x2 = int(round(max(0.0, min(float(raw[2]), image_width))))
        y2 = int(round(max(0.0, min(float(raw[3]), image_height))))
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), max(y1, y2)
        w = max(1, right - left)
        h = max(1, bottom - top)

        box = {
            "id": det.get("id") or f"blk-{idx + 1:04d}",
            "index": idx,
            "label": class_name,
            "kind": kind,
            "class_id": det.get("class_id"),
            "confidence": float(score),
            "bbox": [left, top, left + w, top + h],
            "left": left,
            "top": top,
            "width": w,
            "height": h,
        }
        # Pass through OCR-derived font fields when run_ocr.py has enriched
        # the layout JSON. These are consumed by _font_size_hints below.
        for key in ("n_lines", "line_height_px", "font_size_px",
                    "text", "texts"):
            if key in det:
                box[key] = det[key]
        boxes.append(box)

    boxes.sort(key=lambda b: (b["top"], b["left"]))
    for i, b in enumerate(boxes):
        b["position"] = i
    return boxes

def scaffold_block_html(
    box: dict[str, Any],
    ref_image: Image.Image,
    allow_image_crops: bool,
    assets_dir: Path,
) -> str:
    common = (
        f'id="{box["id"]}" class="block {box["kind"]}" '
        f'data-label="{box["label"]}" data-position="{box["position"]}" '
        f'style="left:{box["left"]}px;top:{box["top"]}px;'
        f'width:{box["width"]}px;height:{box["height"]}px;"'
    )

    if box["kind"] == "image-like" and allow_image_crops:
        asset_name = f"{box['id']}.png"
        save_crop_asset(ref_image, box["bbox"], assets_dir / asset_name)
        src = f"assets/{asset_name}"
        return (
            f'<div {common}>'
            f'<img alt="{box["label"]}" src="{src}" />'
            '</div>'
        )

    return f'<div {common}><div class="content"></div></div>'


def build_scaffold_html(
    boxes: list[dict[str, Any]],
    width: int,
    height: int,
    ref_image: Image.Image,
    allow_image_crops: bool,
    assets_dir: Path,
) -> str:
    blocks = "\n    ".join(
        scaffold_block_html(b, ref_image, allow_image_crops, assets_dir)
        for b in boxes
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width={width}, initial-scale=1" />
  <title>Poster Reconstruction</title>
  <style>
    html, body {{
      margin: 0; padding: 0;
      width: {width}px; height: {height}px;
      overflow: hidden; background: white;
      box-sizing: border-box;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    #canvas {{
      position: relative;
      width: {width}px; height: {height}px;
      overflow: hidden; background: white;
      font-family: "Arial", "Helvetica", sans-serif;
    }}
    .block {{
      position: absolute; overflow: hidden;
    }}
    .block img {{
      display: block; width: 100%; height: 100%; object-fit: cover;
    }}
    .text-like .content, .table-like .content,
    .formula-like .content, .other .content {{
      width: 100%; height: 100%;
      white-space: pre-wrap; overflow: hidden;
      word-break: break-word; line-height: 1.2; color: #111;
    }}
  </style>
  <style id="dynamic-style"></style>
</head>
<body>
  <div id="canvas">
    {blocks}
  </div>
</body>
</html>
"""

def _font_size_hints(boxes: list[dict[str, Any]]) -> str:
    """Build per-block font-size hints for the prompt.

    Prefers the OCR-measured value (from run_ocr.py) when available — emits a
    single deterministic px value. Falls back to a heuristic range derived
    from the bbox height when OCR data is absent.
    """
    lines = []
    for b in boxes:
        if b["kind"] != "text-like":
            continue

        font_px = b.get("font_size_px")
        n_lines = b.get("n_lines")

        if font_px is not None:
            n_str = f", n_lines={n_lines}" if n_lines is not None else ""
            lines.append(
                f'  {b["id"]} ({b["label"]}): {b["width"]}x{b["height"]}px'
                f'{n_str} -> font-size {float(font_px):.1f}px  [measured by OCR]'
            )
            continue

        h = b["height"]
        if h <= 20:
            est = max(8, int(h * 0.8))
        elif h <= 40:
            est = max(10, int(h * 0.72))
        else:
            est = max(12, min(int(h * 0.38), 40))
        min_est = max(8, int(est * 0.72))
        lines.append(
            f'  {b["id"]} ({b["label"]}): '
            f'{b["width"]}x{b["height"]}px -> target font-size about {min_est}-{est}px'
        )
    return "\n".join(lines) if lines else "(none)"


def _dimension_css_hints_block() -> str:
    """Build a per-dimension CSS hints reference string for the prompt."""
    lines: list[str] = []
    for d in _get_dimensions():
        lines.append(f"- **{d.title}** (`{d.id}`): {d.css_hints}")
    return "\n".join(lines)


def build_prompt(
    width: int,
    height: int,
    boxes: list[dict[str, Any]],
    prompt_template_path: Path | None = None,
    design_spec: dict[str, Any] | None = None,
) -> str:
    tpl_path = prompt_template_path or PROMPT_TEMPLATE
    if tpl_path.exists():
        template = tpl_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Prompt template not found: {tpl_path}")

    summary = [
        {"id": b["id"], "label": b["label"], "kind": b["kind"],
         "position": b["position"], "bbox": b["bbox"],
         "width": b["width"], "height": b["height"]}
        for b in boxes
    ]

    if design_spec is None:
        design_block = "(none)"
    else:
        design_block = json.dumps(design_spec, ensure_ascii=False, indent=2)

    replacements = {
        "{width}": str(width),
        "{height}": str(height),
        "{layout_json}": json.dumps(summary, ensure_ascii=False, indent=2),
        "{font_hints}": _font_size_hints(boxes),
        "{design_spec}": design_block,
        "{dimension_css_hints}": _dimension_css_hints_block(),
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result

def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def extract_html(response_text: str) -> str:
    text = strip_think_blocks(response_text).strip()
    fenced = re.findall(r"```(?:html)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced[0].strip()

    for marker_fn in [
        lambda t: t.lower().find("<!doctype"),
        lambda t: t.lower().find("<html"),
    ]:
        idx = marker_fn(text)
        if idx >= 0:
            return text[idx:].strip()

    if "<body" in text.lower() or "<div" in text.lower():
        return (
            '<!doctype html>\n<html lang="en">\n'
            "<head><meta charset=\"utf-8\" /></head>\n"
            f"<body>{text}</body>\n</html>"
        )
    raise ValueError("Model response does not contain parseable HTML")

def find_chrome_binary(preferred: str | None = None) -> str | None:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += ["google-chrome", "google-chrome-stable", "chromium",
                    "chromium-browser", "microsoft-edge"]
    for c in candidates:
        resolved = shutil.which(c) if not os.path.isabs(c) else c
        if resolved and Path(resolved).exists():
            return resolved
    return None


def _ensure_screenshot_size(output_png: Path, width: int, height: int) -> bool:
    """Resize the rendered screenshot to exactly width×height if it differs."""
    if not output_png.exists():
        return False
    img = Image.open(output_png)
    if img.size == (width, height):
        return False
    print(f"  Rendered size {img.size} ≠ expected ({width},{height}), resizing …")
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas.save(output_png)
    return True


def render_html_playwright(
    html_path: Path, output_png: Path, width: int, height: int,
    allow_fallback: bool, chrome_path: str | None,
) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("playwright not installed. Run: pip install playwright && playwright install chromium") from exc

    logs: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, executable_path=os.environ.get("RAP_CHROMIUM") or None)
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.on("console", lambda msg: logs.append(f"console:{msg.type}:{msg.text}"))
            page.goto(html_path.as_uri(), wait_until="networkidle")
            page.add_style_tag(content="*{animation:none!important;transition:none!important;}")
            page.evaluate("()=>{if(document.fonts&&document.fonts.ready)return document.fonts.ready.then(()=>true);return true;}")
            page.screenshot(path=str(output_png), full_page=False)
            browser.close()
        resized = _ensure_screenshot_size(output_png, width, height)
        return {"renderer": "playwright-chromium", "console_logs": logs, "post_resized": resized}
    except Exception as exc:
        if not allow_fallback:
            raise
        binary = find_chrome_binary(chrome_path)
        if not binary:
            raise RuntimeError("No Chrome binary found for fallback") from exc
        output_png.parent.mkdir(parents=True, exist_ok=True)
        cmd = [binary, "--headless=new", "--disable-gpu", "--no-sandbox",
               "--hide-scrollbars", "--allow-file-access-from-files",
               "--force-device-scale-factor=1",
               f"--window-size={width},{height}", f"--screenshot={output_png}",
               html_path.as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Chrome fallback failed: {proc.stderr[:2000]}") from exc
        resized = _ensure_screenshot_size(output_png, width, height)
        return {"renderer": "system-chrome", "playwright_error": str(exc), "post_resized": resized}

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    except Exception:
        return None


def evaluate_images(original: Image.Image, rendered: Image.Image) -> dict[str, Any]:
    a = np.asarray(original.convert("RGB"), dtype=np.float32)
    b = np.asarray(rendered.convert("RGB"), dtype=np.float32)
    if a.shape[:2] != b.shape[:2]:
        rendered_resized = rendered.resize((a.shape[1], a.shape[0]), Image.Resampling.BILINEAR)
        b = np.asarray(rendered_resized.convert("RGB"), dtype=np.float32)

    diff = a - b
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0 else float(20 * math.log10(255) - 10 * math.log10(mse))
    ssim = compute_ssim(a.astype(np.uint8), b.astype(np.uint8))

    diff_gray = np.mean(np.abs(diff), axis=2)
    norm = np.clip(diff_gray, 0, 255).astype(np.uint8)
    heatmap = np.stack([norm, np.zeros_like(norm), 255 - norm], axis=2)

    return {
        "mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim,
        "heatmap": Image.fromarray(heatmap, mode="RGB"),
    }


def load_boxes(layout_json: Path, width: int, height: int) -> list[dict[str, Any]]:
    data = json.loads(layout_json.read_text(encoding="utf-8"))
    entry = data[0] if isinstance(data, list) else data
    return normalize_layout(entry.get("detections", []), width, height, CONF_THRES)


def run_stage4(image_path: Path, layout_json: Path, design_json: Path | None, out_dir: Path, backend: MLLM,
               render: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_file = out_dir / "generated.html"
    image = Image.open(image_path).convert("RGB"); W, H = image.size
    boxes = load_boxes(layout_json, W, H)
    (out_dir / "layout.json").write_text(json.dumps(boxes, indent=2, ensure_ascii=False), encoding="utf-8")
    if design_json and design_json.exists():
        shutil.copy(design_json, out_dir / "design_free.json")
    if not html_file.exists():
        assets_dir = out_dir / "assets"; assets_dir.mkdir(exist_ok=True)
        scaffold = build_scaffold_html(boxes, W, H, image, True, assets_dir)
        (out_dir / "scaffold.html").write_text(scaffold, encoding="utf-8")
        design_spec = json.loads(design_json.read_text(encoding="utf-8")) if design_json and design_json.exists() else None
        if design_spec is not None:
            design_spec = {k: v for k, v in design_spec.items() if not k.startswith("_")}
        prompt = build_prompt(W, H, boxes, design_spec=design_spec)
        full_user_text = prompt + "\n\nScaffold HTML:\n" + scaffold
        response = backend.call(SYSTEM_PROMPT, full_user_text, image, MAX_TOKENS_HTML, label=f"html/{image_path.stem}")
        (out_dir / "model_response.txt").write_text(response, encoding="utf-8")
        html_file.write_text(extract_html(response), encoding="utf-8")
        if not (out_dir / "reference_image.png").exists():
            image.save(out_dir / "reference_image.png")
        (out_dir / "run_config.json").write_text(json.dumps({
            "model": backend.model, "image_size": [W, H], "max_tokens": MAX_TOKENS_HTML, "temperature": TEMPERATURE,
            "conf_thres": CONF_THRES, "n_blocks": len(boxes)}, indent=2), encoding="utf-8")
    if render and not (out_dir / "rendered.png").exists():
        info = render_html_playwright(html_file, out_dir / "rendered.png", W, H, allow_fallback=True, chrome_path=None)
        metrics = evaluate_images(image, Image.open(out_dir / "rendered.png"))
        metrics["heatmap"].save(out_dir / "diff_heatmap.png"); metrics.pop("heatmap")
        metrics.update(info); (out_dir / "render_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return html_file
