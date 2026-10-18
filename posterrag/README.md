# PosterRAG — retrieval-augmented poster generation

![PosterRAG pipeline](../figures/rag_pipeline.png)

PosterRAG treats RAP-25K as a reusable design library. **Stage 1** retrieves the human-crafted template that best fits a
paper; **Stage 2** refills the template's typed slots with the paper's content and repairs residual visual defects, never
touching the template's layout, typography or colours.

| Agent | Module | What it does |
|---|---|---|
| ParserAgent | `agents/parser.py` | PDF (docling) or LaTeX project → structured representation `P_meta`: metadata, abstract, sections, figures, tables, formulas; taxonomy labels `P_d / P_t / P_m` from the RAP-25K vocabulary; introduction summary `P_s` (motivation, problem, key insight, contributions) |
| RetrievalAgent-1 | `agents/retrieval.py` | categorical filtering on (domain, task, method), relaxed stepwise to domain+task, domain+method, domain |
| RetrievalAgent-2 | `agents/retrieval.py` | narrative matching: TF-IDF shortlist of introduction summaries, LLM re-ranking → top-3 |
| RetrievalAgent-3 | `agents/retrieval.py` | aesthetic selection: each candidate poster scored 1–5 on content density, visualization ratio and scientific aesthetics; nearest the neutral midpoint, or to targets/weights derived from `--preferences` |
| MappingAgent | `agents/mapping.py` | slot analysis (id, type, label, position, size, text capacity) → LLM mapping with actions replace_text / replace_image / replace_rendered / keep / clear → deterministic filling; unmapped image slots are hidden |
| ReviewerAgent | `agents/reviewer.py` | see-then-fix: render with headless Chromium, VLM returns a score and block-level CSS patches, applied under safety constraints; stops at score ≥ threshold, max rounds, or no patches |

`template_pool.py` builds the retrieval index from `annotations/all.jsonl` of the dataset and fetches a chosen
template from its ~450 MB tar shard on demand, so the full dataset is never downloaded.

## Setup

```bash
pip install -r posterrag/requirements.txt
playwright install chromium
export ANTHROPIC_API_KEY=...        # or: OPENAI_API_KEY [+ OPENAI_BASE_URL]   or: GEMINI_API_KEY
```

Keys are read from the environment only. Backends: `anthropic` (default model `claude-opus-4-6`, the one used in the
paper), `openai` (any OpenAI-compatible endpoint; `OPENAI_MODEL`), `gemini` (`GEMINI_MODEL`). The same model is used
for text and vision calls.

## Run

```bash
python -m posterrag.run_posterrag --pdf paper.pdf --out outputs/my_poster
python -m posterrag.run_posterrag --pdf paper.pdf --out outputs/my_poster --preferences "figure-heavy, light density"
python -m posterrag.run_posterrag --pdf project.zip --out outputs/my_poster          # LaTeX source instead of PDF
python -m posterrag.run_posterrag --pdf paper.pdf --out outputs/x --template ICLR_2025_5006   # fixed template
```

Outputs: `parse/` (parsed content, `pmeta.json`), `retrieval.json` (candidates, shortlist, aesthetic scores),
`mapping/` (slot mappings, filled HTML), `review/` (per-round screenshots and patches), `poster.html`, `poster.png`,
`summary.json`. Templates and posters are cached under `~/.cache/rap` (`RAP_CACHE`). With a local copy of the dataset
use `--local-root /path/to/RAP`.

## Notes

- Retrieval uses the v3 taxonomy carried by the public annotations (15 domains / 20 tasks / 39 methods); the
  ParserAgent classifies new papers into that same vocabulary, so categorical filtering always has the right labels.
- The release follows the paper's two-stage description. Steps that existed in our experimental code but are not part
  of the method in the paper (institution-logo search, a font-size optimisation pass, an extra LLM content-adaptation
  pass) are not included; font sizes are only ever changed upwards by the ReviewerAgent, as the paper states.
- Formula crops are optional (`--formulas`) because they need the LayoutLMv3 layout environment from `inference/`.
- `latex/` and the slot parsing / filling logic derive from Paper2Poster (MIT, `LICENSE-Paper2Poster`).
