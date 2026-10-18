"""Derender a poster image into a RAP-25K style HTML template.

    python -m derender.run_derender --image poster.png --out out/            # all four stages + render
    python -m derender.run_derender --image-dir posters/ --out out/ --backend openai --model-html gpt-5
    python -m derender.run_derender --image poster.png --out out/ --stages 12  # CPU stages only

Stages: 1 Layout Detection (LayoutLMv3, runs in the layout environment named by RAP_LAYOUT_PYTHON),
2 Font Size Estimation (PaddleOCR; RAP_OCR_PYTHON to use another interpreter), 3 Design Captioning (MLLM),
4 HTML Reconstruction (MLLM) + render. Every stage is skipped when its output already exists, so a batch can be
resumed. Keys are read from the environment only (see mllm.py / README.md).
Output per poster: <out>/<stem>/{01_layout, 02_ocr, 03_design_free, 04_html}; 04_html is the template folder
(generated.html, assets/, rendered.png, layout.json, design_free.json, run_config.json).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _run(cmd: list[str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[1]} failed (exit {r.returncode}); see {log}")


def derender(image: Path, out: Path, stages: str, backend_name: str, model_design: str | None,
             model_html: str | None, render: bool = True) -> dict:
    t0 = time.time(); out.mkdir(parents=True, exist_ok=True); log = out / "derender.log"
    d1, d2, d3, d4 = out / "01_layout", out / "02_ocr", out / "03_design_free", out / "04_html"
    layout = d1 / "layout_results.json"; fonts = d2 / "layout_with_font.json"; design = d3 / "design_free.json"
    if "1" in stages and not layout.exists():
        _run([os.environ.get("RAP_LAYOUT_PYTHON", sys.executable), str(HERE / "stage1_layout.py"),
              "--input", str(image), "--output_dir", str(d1)], log)
    if "2" in stages and not fonts.exists():
        _run([os.environ.get("RAP_OCR_PYTHON", sys.executable), str(HERE / "stage2_font.py"),
              "--input", str(image), "--layout_json", str(layout), "--output_dir", str(d2)], log)
    if "3" in stages or "4" in stages:
        from .mllm import make_backend
        if "3" in stages and not design.exists():
            from .stage3_caption import run_stage3
            run_stage3(image, d3, make_backend(backend_name, model_design))
        if "4" in stages:
            from .stage4_html import run_stage4
            run_stage4(image, fonts if fonts.exists() else layout, design if design.exists() else None, d4,
                       make_backend(backend_name, model_html), render=render)
    done = {k: p.exists() for k, p in (("layout", layout), ("fonts", fonts), ("design", design),
                                        ("html", d4 / "generated.html"), ("render", d4 / "rendered.png"))}
    return {"image": str(image), "out": str(out), "seconds": round(time.time() - t0, 1), **done}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="one poster image"); g.add_argument("--image-dir", help="directory of poster images")
    ap.add_argument("--out", required=True, help="output root; one sub-folder per poster")
    ap.add_argument("--stages", default="1234", help="subset of 1234 to run (default all)")
    ap.add_argument("--backend", default=os.environ.get("RAP_MLLM_BACKEND", "anthropic"), choices=["anthropic", "openai", "mock"])
    ap.add_argument("--model-design", default=None, help="model for Stage 3 (default: backend default)")
    ap.add_argument("--model-html", default=None, help="model for Stage 4 (default: backend default)")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    images = [Path(a.image)] if a.image else sorted(p for p in Path(a.image_dir).iterdir() if p.suffix.lower() in IMAGE_EXT)
    for i, img in enumerate(images, 1):
        try:
            r = derender(img, Path(a.out) / img.stem, a.stages, a.backend, a.model_design, a.model_html, not a.no_render)
            print(f"[{i}/{len(images)}] {img.name}: " + ", ".join(f"{k}={'ok' if v else '-'}" for k, v in r.items() if isinstance(v, bool)) + f" ({r['seconds']} s)", flush=True)
        except Exception as e:                                             # noqa: BLE001
            print(f"[{i}/{len(images)}] {img.name}: FAILED {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
