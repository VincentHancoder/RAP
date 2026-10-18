# RAP-150 benchmark

The held-out benchmark of the RAP paper: **150 papers, 10 per domain of the 15-domain taxonomy**, each paired with its
human-crafted poster. These papers are part of RAP-25K, so their posters, PDFs and derendered templates are in the dataset
repository; this folder holds the exact model inputs we used, so a run of `RAP-Qwen3.8-27B-RL` can be reproduced
token for token.

| File | Content |
|---|---|
| `prompts.jsonl` | one row per paper: `paper_id`, `title`, `domain`, `messages` (system + user turn) and `images` (thumbnail paths, interleaved at the `<image>` markers of the user turn) |
| `thumbnails/<paper_id>/*.jpg` | the figure/table/formula thumbnails the model sees (sized to the training budget) |
| `ids.txt`, `meta.jsonl` | the 150 paper ids with title, domain and number of assets |

The user turn is the training format: `<paper>` (title, venue, authors, body in reading order from the LayoutLMv3 page
parse), `<assets>` (asset ids with kind, aspect ratio, page, section and caption), the thumbnails, `<instruction>`
(a short request verbalised from the reference poster's measured attributes and design caption) and `<layout_spec>`
(the structural specification derived from the reference poster's template). `ICML_2025_282` is a position paper
without figures or tables; its prompt lists no assets.

Full-resolution asset crops (needed to render the final poster from the generated HTML) are on the dataset repository:
`benchmark/rap_test150_assets.tar` in `VincentHancoder/RAP` (143 MB; member `<paper_id>/assets/<asset_id>.png`).
Reference posters: `posters/<Conference>_<Year>/<paper_id>.{png,jpg}` in the same repository.

```bash
# full-resolution assets
python - <<'PY'
from huggingface_hub import hf_hub_download; import tarfile
p = hf_hub_download("VincentHancoder/RAP", "benchmark/rap_test150_assets.tar", repo_type="dataset")
tarfile.open(p).extractall("benchmark/rap_test150/assets")
PY
# generate all 150 posters and render them
python -m inference.generate benchmark --out outputs/rap_test150 --assets benchmark/rap_test150/assets
```
