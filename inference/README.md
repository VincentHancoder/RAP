# Inference with RAP-Qwen3.8-27B-RL

`RAP-Qwen3.8-27B-RL` is Qwen3.8-27B post-trained on RAP-25K (full-parameter SFT, then full-parameter GRPO). It reads the
paper text, the extracted figure/table/formula assets and a design instruction, and writes the poster as one
self-contained HTML page. This folder contains the paper parser used in training, the generation driver and the renderer.

## Setup

Two Python environments, because the layout detector (LayoutLMv3 from PDF-Extract-Kit, detectron2) needs an older
Transformers than the model:

```bash
# 1. main environment: the model, PDF parsing, rendering
pip install -r inference/requirements.txt
playwright install chromium            # or set PATH to a chromium binary

# 2. layout environment (Stage 1 of the paper parser); create it yourself ...
python -m venv ~/rap-layout && ~/rap-layout/bin/pip install -r inference/requirements-layout.txt
export RAP_LAYOUT_PYTHON=~/rap-layout/bin/python
# ... or skip this step: without RAP_LAYOUT_PYTHON the parser creates .runtime/layout_env on first use.
```

Weights are downloaded from the Hub on first use: the model (`VincentHancoder/RAP-Qwen3.8-27B-RL`, 55 GB BF16) and
the layout detector (`opendatalab/PDF-Extract-Kit-1.0`, `models/Layout/LayoutLMv3/model_final.pth`). The model needs
about 60 GB of GPU memory in BF16 (one 80 GB GPU, or two 48 GB GPUs with `device_map="auto"`).

## Reproduce the RAP-150 benchmark

```bash
python - <<'PY'                     # full-resolution assets for rendering (143 MB)
from huggingface_hub import hf_hub_download; import tarfile
tarfile.open(hf_hub_download("VincentHancoder/RAP", "benchmark/rap_test150_assets.tar", repo_type="dataset")).extractall("benchmark/rap_test150/assets")
PY
python -m inference.generate benchmark --out outputs/rap_test150 --assets benchmark/rap_test150/assets
```

This runs the 150 prompts in `benchmark/rap_test150/prompts.jsonl` exactly as in the paper: greedy decoding,
`repetition_penalty 1.1`, up to 12,288 new tokens, `enable_thinking=False`. Each paper gets
`outputs/rap_test150/<paper_id>/{model_output.txt, poster.html, poster.png, info.json}`. Resume by re-running; finished
papers are skipped. `--ids NeurIPS_2023_72958 ...` runs a subset.

## A new paper

```bash
python -m inference.generate pdf --pdf paper.pdf --out outputs/my_poster \
    --instruction "Create a clear academic poster ..." --canvas "Landscape · 3:2" --columns 3 --accent "#584287"
```

The parser (`pdf_pipeline.prepare`) renders the first 12 pages at 200 dpi, detects figures/tables/formulas with
LayoutLMv3, reads captions and section roles (`paper_sections.py`, `paper_side.py`), budgets the text to the training
length and builds the same `<paper> / <assets> / <instruction> / <layout_spec>` prompt as the training data. `--spec`
replaces the automatically derived layout specification. Canvases: `Landscape · 3:2`, `Landscape · 16:9`,
`Portrait · 2:3`, `Square`.

`poster_render.py` resolves `<img src="asset_id">` to the offered assets only, strips scripts and remote references,
renders with headless Chromium and writes `poster.png`, `poster.html`, `model_output.txt` and a zip bundle.

## Files

| File | Purpose |
|---|---|
| `generate.py` | benchmark and single-PDF generation |
| `pdf_pipeline.py` | prompt construction (system prompt, paper block, asset list, thumbnails, instruction, layout spec) |
| `paper_sections.py`, `paper_side.py` | section roles and text budgets; asset registry, captions, reading order |
| `layout_runtime.py`, `layout_worker.py`, `layout_vendor/` | Stage-1 layout detection in its own environment (PDF-Extract-Kit LayoutLMv3; see `../docs/THIRD_PARTY_NOTICES.md`) |
| `poster_render.py`, `render_html.py` | safe HTML assembly and PNG rendering |
