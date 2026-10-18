# Poster derendering

![Derendering pipeline](../figures/derender_pipeline.png)

Turns a poster image into the template format shipped in RAP-25K: `generated.html` with cropped `assets/`,
`rendered.png`, `layout.json` and `design_free.json`.

| Stage | Module | What it does |
|---|---|---|
| 1 Layout Detection | `stage1_layout.py` | LayoutLMv3 (PDF-Extract-Kit) localizes title, text, figure, table, formula and caption regions with confidence scores → `layout_results.json` |
| 2 Font Size Estimation | `stage2_font.py` | PaddleOCR reads each text-like box; OCR boxes are grouped into lines by vertical overlap ≥ 0.5; `font_size = (box_height / n_lines) × 0.8` → `layout_with_font.json` |
| 3 Design Captioning | `stage3_caption.py` | an MLLM sees only the poster and returns a strict JSON design caption: global palette / typography / decorations and 3–7 regions, each with nine design dimensions plus notes → `design_free.json` |
| 4 HTML Reconstruction | `stage4_html.py` | an HTML scaffold with absolutely positioned blocks (image-like regions cropped and embedded, text-like regions as empty slots with their measured font size) plus the poster and the design caption go to an MLLM, which writes the text and the CSS → `generated.html`; rendered with headless Chromium and compared with the original (`render_metrics.json`, `diff_heatmap.png`) |

The prompts and settings are the ones used to build RAP-25K (Claude Opus 4.6, temperature 0.2, 16K / 32K output
tokens, images sent at ≤ 1568 px).

## Setup

```bash
pip install -r derender/requirements.txt
playwright install chromium
# Stage 1 needs detectron2 + an older Transformers; keep it in its own environment:
python -m venv ~/rap-layout && ~/rap-layout/bin/pip install -r inference/requirements-layout.txt
export RAP_LAYOUT_PYTHON=~/rap-layout/bin/python
```

LayoutLMv3 weights (`opendatalab/PDF-Extract-Kit-1.0`) and the PaddleOCR PP-OCRv4 models are downloaded on first
use. Set `RAP_OCR_PYTHON` if PaddleOCR lives in yet another interpreter.

Keys come from the environment only:

```bash
export ANTHROPIC_API_KEY=...                       # backend anthropic (default), optional ANTHROPIC_BASE_URL
export OPENAI_API_KEY=... OPENAI_BASE_URL=...      # backend openai: OpenAI, OpenRouter, vLLM, any compatible server
```

## Run

```bash
python -m derender.run_derender --image poster.png --out outputs/derender            # all four stages
python -m derender.run_derender --image-dir posters/ --out outputs/derender --backend openai --model-html <model>
python -m derender.run_derender --image poster.png --out outputs/derender --stages 12  # layout + fonts only
python -m derender.run_derender --image poster.png --out outputs/derender --backend mock  # exercise the pipeline offline
```

Output per poster: `<out>/<stem>/01_layout/`, `02_ocr/`, `03_design_free/`, `04_html/` — the last one is the template
folder. Stages whose outputs exist are skipped, so a batch can be resumed.

The layout-detector code in `../inference/layout_vendor/` is from PDF-Extract-Kit (AGPL-3.0); see
`../docs/THIRD_PARTY_NOTICES.md`.
