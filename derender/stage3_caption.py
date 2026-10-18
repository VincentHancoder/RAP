"""Stage 3 — Design Captioning.

The poster image alone is shown to a multimodal LLM, which returns a strict JSON design caption: a global
caption, page-level decorations / palette / typography, and 3–7 semantic regions each described with the
nine design dimensions (composition, visual anchor, reading flow, layer interaction, density, colour scheme,
typography, emphasis, decoration) plus free-form notes. Output: `design_free.json` (+ the raw response).
Prompt and post-processing are the ones used to build RAP-25K.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .design_dimensions import DIMENSIONS, Dimension, REGION_DIM_IDS, GLOBAL_DECORATION_DIM_ID, TYPOGRAPHY_DIM_ID
from .mllm import MLLM, MAX_TOKENS_DESIGN


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _allowed_enums() -> dict[str, list[str]]:
    return {d.id: list(d.variants) for d in DIMENSIONS}


def _dim_lookup() -> dict[str, Dimension]:
    return {d.id: d for d in DIMENSIONS}


# ===== Prompt construction =================================================

SYSTEM_PROMPT = (
    "You are a senior visual designer and information architect specialised "
    "in academic posters. Your job is to perceive the structural and "
    "stylistic design of a reference poster image and return a strict JSON "
    "description that downstream code can render back to HTML faithfully."
)


def _format_dimension_block(dim: Dimension) -> str:
    lines: list[str] = []
    lines.append(f"### {dim.title}  (`{dim.id}`)  [level={dim.level}]")
    lines.append(dim.definition.strip())
    lines.append("Allowed values:")
    for v in dim.variants:
        desc = dim.variant_descriptions.get(v)
        if desc:
            lines.append(f"  - `{v}` -- {desc}")
        else:
            lines.append(f"  - `{v}`")
    lines.append(f"CSS guidance: {dim.css_hints.strip()}")
    return "\n".join(lines)


def _build_schema_example() -> dict[str, Any]:
    return {
        "global_caption": (
            "Multi-sentence text. Cover (a) paper topic + which semantic "
            "sections appear on the poster, AND (b) overall design language: "
            "palette feel, typography character, layout rhythm, decorative "
            "motifs, density, signature visual moves."
        ),
        "global": {
            "decorations": ["<one or more variants of global_decorations>"],
            "decorations_notes": "Concrete sentence about page-level decorations.",
            "palette": {
                "surface": "#RRGGBB",
                "primary": "#RRGGBB",
                "accent": "#RRGGBB",
                "fg": "#RRGGBB",
            },
            "typography_family": "<one variant of typography>",
            "typography_notes": (
                "Concrete sentence about type family and weight hierarchy "
                "(e.g. bold sans title, regular body). "
                "Do NOT mention specific px/pt sizes."
            ),
        },
        "regions": [
            {
                "id": "R-01",
                "type": "header|motivation|method|results|conclusion|figure|footer|...",
                "title": (
                    "The exact title text visible in this region "
                    "(e.g. 'Motivation', 'Our Method: DiffPoster', "
                    "'Results on COCO'). Write '(no title)' if none."
                ),
                "reading_order": 1,
                "caption": (
                    "3-4 sentences describing all content in this region: "
                    "what text, figures, tables, equations are present and "
                    "what they convey. Do NOT discuss design."
                ),
                **{
                    dim_id: {
                        "value": f"<one variant of {dim_id}>",
                        "notes": (
                            "Concrete one-sentence description of how this "
                            "dimension actually looks in this region (refer "
                            "to colors, paddings, alignments, weight "
                            "hierarchy, etc.; do NOT just restate the "
                            "variant name)."
                        ),
                    }
                    for dim_id in REGION_DIM_IDS
                },
                "design_caption": (
                    "3-5 sentences comprehensively describing the visual "
                    "design of this region: colour palette, typographic "
                    "hierarchy, compositional rhythm, whitespace / density, "
                    "and any decorative treatment. Must NOT restate "
                    "per-dimension variant names."
                ),
            }
        ],
    }


def build_prompt(image_size: tuple[int, int]) -> str:
    W, H = image_size
    dim_section = "\n\n".join(_format_dimension_block(d) for d in DIMENSIONS)
    schema_example = _build_schema_example()

    return f"""\
# Task

Look at the attached reference poster and emit a single JSON document that
captures BOTH the structural decomposition (high-level semantic regions you
identify yourself from the image) AND the stylistic design (per-region
tokens chosen from a fixed dimension vocabulary, plus free-form `notes` and
`design_caption`).

You do NOT receive any pre-computed bounding boxes. Identify regions
entirely from what you observe in the image.

## Canvas

The image is exactly **{W} x {H} px**.

## Design dimension vocabulary

Use these dimensions for per-region tokens (unless tagged `level=global`).
Pick the closest variant for each `value`.

{dim_section}

## Output schema

Return ONE JSON object matching this shape (placeholders shown with angle
brackets are illustrative; replace with your real perceptions):

```json
{json.dumps(schema_example, ensure_ascii=False, indent=2)}
```

## Hard constraints

### Region granularity — CRITICAL

A **region** corresponds to a major section that a human reader would
name when describing the poster (e.g. "Header", "Motivation",
"Method", "Results", "Conclusion"). It is identified by a section
heading or a clearly bounded zone.

**DO NOT split a section into multiple regions** just because it
contains several figures, panels, columns, or sub-blocks. All content
that falls under the same section heading belongs to ONE region.

Correct (coarse, section-level):
  R-01 Header  |  R-02 Motivation  |  R-03 Method  |  R-04 Results

Wrong (too fine — sub-panels turned into separate regions):
  R-01 Header  |  R-02 Motivation  |  R-03 Method  |  R-04 Data examples
  |  R-05 Results figure 1  |  R-06 Results figure 2  |  R-07 Analysis

Use **3 to 7 regions** for a typical academic poster. If in doubt,
merge adjacent similar content into the same region rather than
splitting.

- `id` values must be "R-01", "R-02", … in reading order.
- `type` should be a short semantic label (header / motivation / method /
  results / conclusion / figure / footer / abstract / related_work / …).
- `title`: write the **exact title text** visible at the start of this
  region as it appears in the image (e.g. "Motivation and Summary",
  "Our Method: WCST-ML"). If the region has no visible title heading,
  write "(no title)".
- `reading_order`: 1-based integer; must be consecutive, starting at 1,
  covering every region exactly once.
- `caption`: REQUIRED, 3-4 sentences. Describe ALL concrete content in
  this region (text paragraphs, figures, tables, equations, lists, logos,
  etc.) and what they communicate. Do NOT discuss visual design here.
- Pick `value` strictly from the listed variants of each dimension.
- `notes` is REQUIRED for every dimension entry: write ONE concrete
  sentence describing how this dimension actually looks in this region
  (concrete colors / paddings / alignments / weight hierarchy).
  Never restate the variant name verbatim, never leave empty.
- For `typography` notes, describe ONLY the type family and weight
  hierarchy (e.g. "bold sans-serif section header over regular-weight
  body copy"). Do NOT mention specific pixel or point sizes — font-size
  measurement is handled separately by OCR downstream.
- For `interaction` notes, explicitly describe the layering structure:
  state whether there is a solid-colour background carrier, an offset
  underlay, or nested containers; name the background colour, the title
  bar colour/height if present, and whether an inner white/light box
  sits inside an outer coloured frame. Example: "Deep blue solid-carrier
  bar spans the full region width as a title strip; the content area
  beneath uses a white inner panel inset 8px from the blue outer frame."
- For `decoration` notes, state whether the border/radius treatment is
  applied to (a) each individual content block separately, (b) a single
  section-level container wrapping all blocks, or (c) a nested
  combination (outer rounded container + inner square white box). Also
  name the border colour and corner radius.
- For `color_scheme` notes, map concrete observed colours to the four
  palette roles defined in `global.palette` (primary / accent / surface /
  foreground). Example: "Surface is off-white (#F5F5F5); primary is the
  deep blue used for title bars and borders; accent is the orange used
  for highlighted keywords; foreground is near-black body text."
- `global_caption` MUST be multi-sentence and cover BOTH:
  (a) the paper topic + which semantic sections the poster contains, AND
  (b) the overall design language of the whole poster (palette feel,
  typography character, layout rhythm, decorative motifs, density,
  signature visual moves).
- `global.decorations_notes` is REQUIRED. Describe every page-level
  decorative element: the colour, approximate pixel height, and position
  of any header/footer bars (solid_block_carrier); the colour and
  spacing of any divider lines; whether logos or QR codes are present
  and where. Example: "A full-width dark-blue header bar (≈80px) tops
  the poster; thin grey hairline rules separate each main section; a
  university logo appears in the top-right corner."
- `global.typography_notes` is REQUIRED. Describe the overall type
  family and weight hierarchy used across the whole poster.
- `design_caption`: REQUIRED, 3-5 sentences. Comprehensively describe the
  visual design of this region — colour palette usage, typographic
  hierarchy, compositional rhythm, whitespace / density, and any
  decorative treatment. Must NOT restate per-dimension variant names;
  should read as a designer's detailed mental note.

## Output

Return JSON ONLY: no markdown fences, no commentary, no <think> blocks.
Begin with `{{` and end with `}}`.
"""


# ===== JSON extraction =====================================================


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = strip_think_blocks(text).strip()

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", cleaned,
                        flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced[0].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Model response does not contain a JSON object")
    snippet = cleaned[start: end + 1]

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", snippet)
        return json.loads(repaired)


# ===== Post-processing =====================================================


def _validate_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _ensure_dim_struct(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "value": value.get("value", ""),
            "notes": value.get("notes", "") or "",
        }
    if isinstance(value, str):
        return {"value": value, "notes": ""}
    return {"value": "", "notes": ""}


def postprocess_design_free(
    design: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Validate / normalise the free-form VLM output. Returns (clean, warnings)."""
    warnings: list[str] = []
    enums = _allowed_enums()

    # ---- global_caption ---------------------------------------------------
    if not _validate_text(design.get("global_caption")):
        warnings.append("global_caption is empty or missing.")
        design["global_caption"] = design.get("global_caption") or ""

    # ---- global -----------------------------------------------------------
    g = design.setdefault("global", {})
    g.setdefault("decorations", [])
    if not isinstance(g["decorations"], list):
        warnings.append("global.decorations is not a list; coerced to [].")
        g["decorations"] = []
    bad_decor = [
        v for v in g["decorations"]
        if v not in enums.get(GLOBAL_DECORATION_DIM_ID, [])
    ]
    if bad_decor:
        warnings.append(
            f"global.decorations contains values outside enum: {bad_decor}"
        )
    if not _validate_text(g.get("decorations_notes")):
        warnings.append("global.decorations_notes is empty or missing.")
        g["decorations_notes"] = g.get("decorations_notes") or ""

    palette = g.setdefault("palette", {})
    for role in ("surface", "primary", "accent", "fg"):
        if role not in palette:
            warnings.append(f"global.palette.{role} missing.")
            palette[role] = ""

    if not _validate_text(g.get("typography_family")):
        warnings.append("global.typography_family missing.")
        g["typography_family"] = g.get("typography_family") or ""
    elif g["typography_family"] not in enums.get(TYPOGRAPHY_DIM_ID, []):
        warnings.append(
            f"global.typography_family '{g['typography_family']}' not in enum."
        )
    if not _validate_text(g.get("typography_notes")):
        warnings.append("global.typography_notes is empty or missing.")
        g["typography_notes"] = g.get("typography_notes") or ""

    # ---- regions ----------------------------------------------------------
    regions = design.setdefault("regions", [])
    if not isinstance(regions, list) or not regions:
        warnings.append("regions list is empty or missing.")
        return design, warnings

    cleaned_regions: list[dict[str, Any]] = []
    for idx, region in enumerate(regions):
        region_warns: list[str] = []
        if not isinstance(region, dict):
            warnings.append(f"regions[{idx}] is not a dict; skipped.")
            continue

        rid = region.get("id") or f"R-{idx + 1:02d}"
        region["id"] = rid

        # title
        if not _validate_text(region.get("title")):
            region_warns.append("title is empty or missing; set to '(no title)'.")
            region["title"] = "(no title)"

        # reading_order
        try:
            region["reading_order"] = int(region.get("reading_order", idx + 1))
        except (TypeError, ValueError):
            region["reading_order"] = idx + 1
            region_warns.append("reading_order not an int; auto-numbered.")

        # caption
        if not _validate_text(region.get("caption")):
            region_warns.append("caption is empty.")
            region["caption"] = region.get("caption") or ""

        # per-dimension value + notes
        for dim_id in REGION_DIM_IDS:
            entry = _ensure_dim_struct(region.get(dim_id))
            allowed = enums.get(dim_id, [])
            if entry["value"] not in allowed:
                region_warns.append(
                    f"{dim_id}.value '{entry['value']}' not in enum {allowed}."
                )
            if not _validate_text(entry["notes"]):
                region_warns.append(f"{dim_id}.notes is empty.")
            region[dim_id] = entry

        # design_caption
        if not _validate_text(region.get("design_caption")):
            region_warns.append("design_caption is empty.")
            region["design_caption"] = region.get("design_caption") or ""

        if region_warns:
            warnings.append(f"[{rid}] " + " | ".join(region_warns))
        cleaned_regions.append(region)

    design["regions"] = cleaned_regions
    return design, warnings


def run_stage3(image_path: Path, out_dir: Path, backend: MLLM) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "design_free.json"
    if target.exists():
        return target
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(image.size)
    response = backend.call(SYSTEM_PROMPT, prompt, image, MAX_TOKENS_DESIGN, label=f"design/{image_path.stem}")
    (out_dir / "design_response.txt").write_text(response, encoding="utf-8")
    design, warnings = postprocess_design_free(_extract_json(response))
    design["_meta"] = {"stage": "design_captioning", "model": backend.model, "image_size": list(image.size),
                       "warnings": warnings}
    target.write_text(json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
