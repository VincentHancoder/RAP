"""Design dimension vocabulary used by Stage 3 (Design Captioning) and Stage 4 (HTML Reconstruction).

One global dimension (page-level decorations) and nine per-region dimensions: five layout dimensions
(composition, visual anchor, reading flow, layer interaction, density) and four style dimensions
(colour scheme, typography, emphasis, border/decoration). The CSS hints are handed to the HTML
reconstruction model so that every token has a concrete realisation.
"""
from dataclasses import dataclass, field


@dataclass
class Dimension:
    id: str
    level: str                      # "global" or "region"
    title: str
    definition: str
    variants: list[str]
    css_hints: str
    extra_instructions: str = ""
    variant_descriptions: dict[str, str] = field(default_factory=dict)


DIMENSIONS: list[Dimension] = [
    Dimension(
        id="global_decorations", level="global", title="Global decorations",
        definition=("Page-level decorative scaffolding that sits across the whole poster, independent of any "
                    "single semantic block: framing lines, logo / QR placeholders, and background textures."),
        variants=["solid_block_carrier", "thin_divider_lines", "logo_placeholder", "qr_placeholder", "background_pattern"],
        css_hints=("In scaffold HTML: insert extra <div> elements INSIDE #canvas BEFORE all scaffold block divs. "
                   "solid_block_carrier → position:absolute full-width strip (z-index:0). "
                   "thin_divider_lines → position:absolute 1px-height rule (z-index:3). "
                   "logo_placeholder / qr_placeholder → position:absolute corner block. "
                   "background_pattern → repeating-gradient or ::before on #canvas."),
        variant_descriptions={
            "solid_block_carrier": ("A solid filled rectangle that hosts content (header strip, footer bar). In scaffold HTML: "
                                    "add a position:absolute <div> with z-index:0 spanning the full canvas width at the "
                                    "top/bottom; raise content block divs to z-index:2."),
            "thin_divider_lines": "Hairline horizontal / vertical rules separating regions.",
            "logo_placeholder": "Reserved corner block for an institutional logo.",
            "qr_placeholder": "Reserved square block for a QR code.",
            "background_pattern": "Subtle repeating texture or geometric pattern behind content.",
        },
    ),
    Dimension(
        id="composition", level="region", title="Spatial composition",
        definition=("How the block is laid out internally: grid skeleton, flex flow, or absolute overlap. Decides "
                    "whether new content gets wrapped into a tight container or flows in an open space."),
        variants=["grid_2col", "grid_3col", "flex_row", "flex_col", "absolute_overlap"],
        css_hints=("display:grid with grid-template-columns, display:flex with flex-direction, position:absolute "
                   "with overlapping children."),
    ),
    Dimension(
        id="visual_anchor", level="region", title="Visual anchor",
        definition=("Where the eye lands first. Conveyed via size, weight, contrast, or isolation. Even after "
                    "composition is fixed, an anchor tells the viewer which sub-element is the protagonist."),
        variants=["scale_dominant", "weight_dominant", "contrast_dominant", "isolated_focal"],
        css_hints=("transform:scale, font-size, font-weight, color contrast against background, generous "
                   "surrounding whitespace, z-index."),
        variant_descriptions={
            "scale_dominant": "One element is much larger than its peers.",
            "weight_dominant": "Same size but much heavier font weight or stroke.",
            "contrast_dominant": "Same size but high color contrast against the surroundings.",
            "isolated_focal": "Same size and weight but isolated in whitespace.",
        },
    ),
    Dimension(
        id="flow", level="region", title="Reading flow",
        definition=("The ordering DOM and visual layout impose on the reader: F-pattern, Z-pattern, top-down, or "
                    "strict left-right. Maps Latex section order onto a physical reading path."),
        variants=["z_pattern", "f_pattern", "top_down", "left_right"],
        css_hints=("Order of DOM children, flex-direction, grid-auto-flow, explicit numbered badges to suggest "
                   "reading order."),
    ),
    Dimension(
        id="interaction", level="region", title="Layer interaction",
        definition=("Z-axis stacking relationships between decorative carriers and content: solid carrier, offset "
                    "underlay, partial overlay, glass, or flat inline (no overlap)."),
        variants=["solid_carrier", "offset_underlay", "partial_overlay", "glass_morphism", "flat_inline"],
        css_hints=("In scaffold HTML: add a region background <div> (position:absolute, z-index:0) BEFORE the "
                   "section's block divs; set those block divs to z-index:2. "
                   "solid_carrier: bg div fully covers the region + coloured title strip at top. "
                   "offset_underlay: bg div shifted 4-8px right/down from content bounding box. "
                   "partial_overlay: bg div covers only part of the region (e.g. top half). "
                   "glass_morphism: background:rgba() with backdrop-filter:blur(8px) on the bg div. "
                   "flat_inline: no background div needed."),
        variant_descriptions={
            "solid_carrier": ("Color block fully wraps its content. In scaffold HTML: add a position:absolute background "
                              "<div> (z-index:0) sized to cover all blocks in this region ± 8-16px padding; set those "
                              "block divs to z-index:2. Add a thick coloured title strip (≈30-50px) at the top of the container."),
            "offset_underlay": ("Color block is offset, peeking out from behind the content. In scaffold HTML: shift the "
                                "background <div> 4-8px right and down relative to the content bounding box."),
            "partial_overlay": "Image or text only covers part of the block, breaking the frame.",
            "glass_morphism": "Translucent layer on top of content with blur.",
            "flat_inline": "No overlap at all, content lives on flat background. No extra div needed.",
        },
    ),
    Dimension(
        id="density", level="region", title="Density",
        definition="Cognitive load conveyed by negative space: airy, balanced, or dense packing of the same nominal content.",
        variants=["airy", "balanced", "dense"],
        css_hints="padding, gap, line-height, letter-spacing, max-width to control the ratio of negative space inside the block.",
        extra_instructions=("All three variants must contain the SAME placeholder content (same title text and same "
                            "number of body lines) so the only visible difference is spacing."),
    ),
    Dimension(
        id="color_scheme", level="region", title="Color scheme (role-based)",
        definition=("Role-based color usage: surface (background), primary (brand), accent (highlight), foreground "
                    "(text). Show three palettes built out of these four roles."),
        variants=["neutral_palette", "warm_palette", "cool_palette"],
        css_hints=("Define CSS custom properties --surface, --primary, --accent, --fg per variant card and use them on "
                   "background/border/text."),
        extra_instructions=("Inside each variant card, also render four small swatches labelled surface / primary / "
                            "accent / foreground so the role mapping is visible."),
    ),
    Dimension(
        id="typography", level="region", title="Typography",
        definition=("Type family + weight hierarchy that conveys the poster's tone: modern sans, classic serif, "
                    "technical mono. Each card must show a clear weight hierarchy (h1 / h2 / body / caption). "
                    "Do NOT estimate or mention specific pixel/pt sizes in notes — font size measurement is handled "
                    "separately by OCR."),
        variants=["sans_modern", "serif_classic", "mono_technical"],
        css_hints=("font-family with system fallbacks (sans-serif, serif, monospace), font-weight 800 / 600 / 400, "
                   "letter-spacing, line-height."),
    ),
    Dimension(
        id="emphasis", level="region", title="Emphasis strategy",
        definition="How to elevate a keyword or a takeaway line within otherwise uniform body copy.",
        variants=["bold_caps", "color_tint", "icon_prefix", "highlight_bg"],
        css_hints=("font-weight, text-transform:uppercase, color, ::before with a glyph or shape, background-color "
                   "highlight wrapper."),
    ),
    Dimension(
        id="decoration", level="region", title="Border & decoration",
        definition=("Edge treatment of the block: stroke style, corner radius, elevation. Controls perceived precision "
                    "and material feel."),
        variants=["border_solid_square", "border_dashed_square", "border_solid_rounded", "border_none_shadow", "border_thick_accent"],
        css_hints=("In scaffold HTML: apply border/border-radius/box-shadow to the region background <div> (z-index:0), "
                   "NOT to each individual scaffold block div. "
                   "border_solid_square → border:2px solid <primary>; border-radius:0. "
                   "border_dashed_square → border:2px dashed <primary>; border-radius:0. "
                   "border_solid_rounded → border:2px solid <primary>; border-radius:12-16px. "
                   "border_none_shadow → no border; box-shadow:0 4px 16px rgba(0,0,0,0.15). "
                   "border_thick_accent → border-left:6px solid <accent>; border-top:none. "
                   "If a nested inner white box is present (per design_caption), add a second inner <div> "
                   "(z-index:1, background:#fff, matching border-radius)."),
    ),
]

DIMENSION_INDEX: dict[str, Dimension] = {d.id: d for d in DIMENSIONS}
REGION_DIM_IDS = ["composition", "visual_anchor", "flow", "interaction", "density",
                  "color_scheme", "typography", "emphasis", "decoration"]
GLOBAL_DECORATION_DIM_ID = "global_decorations"
TYPOGRAPHY_DIM_ID = "typography"
