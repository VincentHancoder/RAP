"""
Render LaTeX formulas and tables as PNG images.

Tables: compiled via pdflatex (TinyTeX) -> pdftocairo -> PIL crop.
Equations: rendered via MathJax + Playwright.
"""

import base64
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote


from PIL import Image as _PILImage
_PILImage.MAX_IMAGE_PIXELS = 200_000_000

TINYTEX_PDFLATEX = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux/pdflatex")

MATHJAX_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$','$'],['\\(','\\)']],
    displayMath: [['$$','$$'],['\\[','\\]']],
    packages: {{'[+]': ['ams','boldsymbol','mathtools']}}
  }},
  svg: {{fontCache: 'global'}},
  startup: {{
    pageReady: function() {{
      return MathJax.startup.defaultPageReady().then(function() {{
        document.getElementById('ready-flag').style.display = 'block';
      }});
    }}
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  body {{
    margin: 0;
    padding: {padding}px;
    background: {background};
    font-size: {font_size}px;
    color: {color};
    font-family: 'Computer Modern', 'Latin Modern Math', serif;
    display: inline-block;
  }}
  #content {{
    display: inline-block;
  }}
  #ready-flag {{
    display: none;
  }}
</style>
</head><body>
<div id="content">{content}</div>
<div id="ready-flag"></div>
</body></html>"""


TABLE_TEX_TEMPLATE = r"""\documentclass{{article}}
\usepackage[paperwidth=30cm,paperheight=20cm,margin=1cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{multirow}}
\usepackage{{array}}
\usepackage{{tabularx}}
\pagestyle{{empty}}
\begin{{document}}
\thispagestyle{{empty}}
{content}
\end{{document}}
"""


def _find_pdflatex() -> Optional[str]:
    if os.path.isfile(TINYTEX_PDFLATEX):
        return TINYTEX_PDFLATEX
    for candidate in ["/usr/bin/pdflatex", "/usr/local/bin/pdflatex"]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _render_table_via_pdflatex(
    table_latex: str,
    dpi: int = 300,
) -> Optional[str]:
    """Compile a LaTeX table snippet via pdflatex, convert to cropped PNG."""
    pdflatex = _find_pdflatex()
    if not pdflatex:
        return None

    cleaned = _clean_table_for_compilation(table_latex)

    tex_content = TABLE_TEX_TEMPLATE.format(content=cleaned)

    tmpdir = tempfile.mkdtemp(prefix="latex_table_")
    tex_path = os.path.join(tmpdir, "table.tex")
    pdf_path = os.path.join(tmpdir, "table.pdf")
    png_path = os.path.join(tmpdir, "table.png")

    try:
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
            cwd=tmpdir,
            capture_output=True,
            timeout=30,
        )

        if not os.path.isfile(pdf_path):
            log_path = os.path.join(tmpdir, "table.log")
            if os.path.isfile(log_path):
                with open(log_path, "r", errors="ignore") as f:
                    log = f.read()
                errors = [l for l in log.split("\n") if l.startswith("!")]
                if errors:
                    print(f"    pdflatex errors: {errors[:3]}")
            return None

        subprocess.run(
            ["pdftocairo", "-png", "-r", str(dpi), "-singlefile", pdf_path, png_path.replace(".png", "")],
            capture_output=True,
            timeout=15,
        )

        actual_png = png_path
        if not os.path.isfile(actual_png):
            return None

        from PIL import Image, ImageChops, ImageOps
        img = Image.open(actual_png).convert("RGB")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            cropped = img.crop(bbox)
            cropped = ImageOps.expand(cropped, border=15, fill=(255, 255, 255))
        else:
            cropped = img

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    except subprocess.TimeoutExpired:
        print("    pdflatex timed out")
        return None
    except Exception as e:
        print(f"    pdflatex rendering failed: {e}")
        return None
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _clean_table_for_compilation(latex: str) -> str:
    """Strip commands that TinyTeX's minimal install can't handle."""
    # Fix malformed tabularx: \begin{tabular}{\textwidth}\n{col_spec}
    m = re.search(r"\\begin\{tabular\}\{\\textwidth\}\s*\{([^}]*)\}", latex)
    if m:
        col_spec = m.group(1)
        latex = latex[:m.start()] + f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}" + latex[m.end():]
        latex = latex.replace("\\end{tabular}", "\\end{tabularx}")

    latex = re.sub(r"\\cellcolor(?:\[[^\]]*\])?\{[^}]*\}", "", latex)
    latex = re.sub(r"\\rowcolor(?:\[[^\]]*\])?\{[^}]*\}", "", latex)
    latex = re.sub(r"\\arrayrulecolor\{[^}]*\}", "", latex)
    latex = re.sub(r"\\columncolor\{[^}]*\}", "", latex)
    latex = re.sub(r"\\textcolor\{[^}]*\}\{([^}]*)\}", r"\1", latex)
    latex = re.sub(r"\\cite\{[^}]*\}", "", latex)
    latex = re.sub(r"\\ref\{[^}]*\}", "", latex)
    latex = re.sub(r"\\label\{[^}]*\}", "", latex)
    latex = re.sub(r"\\ding\{51\}", r"$\\checkmark$", latex)
    latex = re.sub(r"\\ding\{55\}", r"$\\times$", latex)
    latex = re.sub(r"\\ding\{\d+\}", "", latex)
    latex = re.sub(r"\\url\{[^}]*\}", "", latex)
    latex = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", latex)
    latex = re.sub(r"\\etal\b", "et al.", latex)
    latex = re.sub(r"\\texttt\{([^}]*)\}", r"\1", latex)
    latex = re.sub(r"\\textsf\{([^}]*)\}", r"\1", latex)
    latex = re.sub(r"\\textsc\{([^}]*)\}", r"\1", latex)
    latex = re.sub(r"\\noalign\{[^}]*\}", "", latex)
    latex = re.sub(r"\\addlinespace(?:\[[^\]]*\])?", "", latex)
    latex = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}\{", "", latex)
    if "\\resizebox" in latex:
        latex = re.sub(r"\}(\s*)$", r"\1", latex, count=1)
    return latex


def _render_latex_to_png(
    latex_content: str,
    width: int = 1200,
    font_size: int = 28,
    padding: int = 20,
    background: str = "white",
    color: str = "black",
) -> Optional[str]:
    """Render LaTeX math content to a PNG image via MathJax + Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Warning: playwright not available for LaTeX rendering")
        return None

    content = f"$${latex_content}$$"

    html = MATHJAX_HTML_TEMPLATE.format(
        content=content,
        font_size=font_size,
        padding=padding,
        background=background,
        color=color,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
    tmp.write(html)
    tmp.close()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": 800})
            path_posix = Path(tmp.name).resolve().as_posix()
            file_url = "file://" + quote(path_posix, safe="/:")
            page.goto(file_url, wait_until="load", timeout=60000)
            page.wait_for_timeout(5000)

            content_el = page.query_selector("#content")
            if content_el:
                screenshot_bytes = content_el.screenshot(type="png")
            else:
                screenshot_bytes = page.screenshot(type="png")

            browser.close()

        b64 = base64.standard_b64encode(screenshot_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print(f"  Warning: LaTeX rendering failed: {e}")
        return None
    finally:
        os.unlink(tmp.name)


def _extract_key_equations(parsed_content: dict, latex_dir: str = None) -> List[dict]:
    """Extract key equations from the parsed LaTeX content or raw .tex files."""
    equations = []

    equation_patterns = [
        re.compile(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", re.DOTALL),
        re.compile(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", re.DOTALL),
        re.compile(r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}", re.DOTALL),
        re.compile(r"\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}", re.DOTALL),
    ]

    raw_sections = parsed_content.get("_raw_sections", [])
    all_text = "\n".join(s.get("content", "") for s in raw_sections)

    if not any(pat.search(all_text) for pat in equation_patterns):
        if latex_dir and os.path.isdir(latex_dir):
            all_text = ""
            for root, _, files in os.walk(latex_dir):
                for fn in files:
                    if fn.endswith(".tex"):
                        try:
                            with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as f:
                                all_text += f.read() + "\n"
                        except Exception:
                            pass

    eq_id = 0
    seen = set()
    for pat in equation_patterns:
        for m in pat.finditer(all_text):
            eq_text = m.group(1).strip()
            eq_text = re.sub(r"\\label\{[^}]*\}", "", eq_text).strip()
            eq_text = re.sub(r"\\nonumber\s*", "", eq_text).strip()
            if len(eq_text) > 10 and eq_text not in seen:
                seen.add(eq_text)
                eq_id += 1
                equations.append({
                    "equation_id": eq_id,
                    "latex": eq_text,
                })

    return equations


def _extract_key_tables(parsed_content: dict, latex_dir: str = None) -> List[dict]:
    """Extract raw LaTeX table content for direct pdflatex compilation."""
    tables = parsed_content.get("tables", [])
    result = []

    all_tex = ""
    if latex_dir and os.path.isdir(latex_dir):
        for root, _, files in os.walk(latex_dir):
            for fn in files:
                if fn.endswith(".tex"):
                    try:
                        with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as f:
                            all_tex += f.read() + "\n"
                    except Exception:
                        pass

    # Extract full tabular environments from raw .tex files
    if all_tex:
        table_pat = re.compile(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.DOTALL)
        raw_tables = list(table_pat.finditer(all_tex))
        for i, table in enumerate(tables):
            if not table.get("content") and i < len(raw_tables):
                block = raw_tables[i].group(1)
                tab_m = re.search(
                    r"(\\begin\{tabular[*x]?\}.*?\\end\{tabular[*x]?\})",
                    block, re.DOTALL,
                )
                if tab_m:
                    table["content"] = tab_m.group(1)

    for table in tables:
        table_id = table.get("table_id", 0)
        caption = table.get("caption", "")
        content = table.get("content", "")

        if not content:
            continue

        if not re.search(r"\\begin\{tabular", content):
            continue

        result.append({
            "table_id": table_id,
            "caption": caption,
            "latex": content,
        })

    return result


def render_equations_and_tables(
    parsed_content: dict,
    canvas_width: int = 4000,
    background: str = "white",
    color: str = "black",
    latex_dir: str = None,
) -> Dict[str, str]:
    """Render all key equations and tables as base64 PNG images."""
    registry = {}

    equations = _extract_key_equations(parsed_content, latex_dir=latex_dir)
    print(f"  Found {len(equations)} equations to render")

    font_size = max(28, canvas_width // 120)
    render_width = min(canvas_width // 2, 1600)

    for eq in equations[:10]:
        key = f"equation_{eq['equation_id']}"
        uri = _render_latex_to_png(
            eq["latex"],
            width=render_width,
            font_size=font_size,
            background=background,
            color=color,
        )
        if uri:
            registry[key] = uri
            print(f"    Rendered {key}: {eq['latex'][:50]}...")

    tables = _extract_key_tables(parsed_content, latex_dir=latex_dir)
    print(f"  Found {len(tables)} tables to render")

    has_pdflatex = _find_pdflatex() is not None
    print(f"  pdflatex available: {has_pdflatex}")

    dpi = min(400, max(300, canvas_width // 8))
    for tab in tables[:5]:
        key = f"table_{tab['table_id']}"
        uri = None
        if has_pdflatex:
            uri = _render_table_via_pdflatex(tab["latex"], dpi=dpi)
            if uri:
                print(f"    Rendered {key} via pdflatex: {tab['caption'][:50]}...")
        if not uri:
            print(f"    FAILED to render {key}: {tab['caption'][:50]}...")
        if uri:
            registry[key] = uri

    return registry
