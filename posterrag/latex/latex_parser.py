"""
Step 1: Parse LaTeX source into structured JSON.

Pure Python regex-based parsing — no TeX compilation needed.
Handles:
  - \input{} / \include{} resolution
  - \title{}, \author{}, abstract extraction
  - \section{} / \subsection{} splitting
  - \includegraphics{} figure reference extraction
  - \begin{figure}...\caption{}...\end{figure} extraction
  - \begin{table}...\caption{}...\end{table} extraction
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_tex(path: str) -> str:
    """Read a .tex file, trying utf-8 then latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def _strip_comments(tex: str) -> str:
    """Remove LaTeX line comments (lines starting with % or inline %)."""
    lines = []
    for line in tex.split("\n"):
        # Remove inline comments (not escaped \%)
        cleaned = re.sub(r"(?<!\\)%.*", "", line)
        lines.append(cleaned)
    return "\n".join(lines)


def _resolve_inputs(tex: str, base_dir: str, depth: int = 0) -> str:
    """
    Recursively resolve \input{file} and \include{file} commands.
    """
    if depth > 10:
        return tex

    def replacer(m):
        fname = m.group(1).strip()
        if not fname.endswith(".tex"):
            fname += ".tex"
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            child_dir = os.path.dirname(fpath)
            child_tex = _read_tex(fpath)
            return _resolve_inputs(child_tex, child_dir, depth + 1)
        return m.group(0)  # keep original if file not found

    tex = re.sub(r"\\input\{([^}]+)\}", replacer, tex)
    tex = re.sub(r"\\include\{([^}]+)\}", replacer, tex)
    return tex


def _extract_braced(tex: str, cmd: str) -> Optional[str]:
    """Extract content of \\cmd{...}, handling nested braces."""
    pattern = re.compile(r"\\" + re.escape(cmd) + r"\s*\{")
    m = pattern.search(tex)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(tex) and depth > 0:
        if tex[i] == "{":
            depth += 1
        elif tex[i] == "}":
            depth -= 1
        i += 1
    return tex[start : i - 1].strip()


def _extract_all_braced(tex: str, cmd: str) -> List[str]:
    """Extract all occurrences of \\cmd{...}."""
    results = []
    pattern = re.compile(r"\\" + re.escape(cmd) + r"\s*\{")
    for m in pattern.finditer(tex):
        start = m.end()
        depth = 1
        i = start
        while i < len(tex) and depth > 0:
            if tex[i] == "{":
                depth += 1
            elif tex[i] == "}":
                depth -= 1
            i += 1
        results.append(tex[start : i - 1].strip())
    return results


def extract_metadata(tex: str) -> dict:
    """Extract title, authors, affiliations from preamble."""
    title = _extract_braced(tex, "title") or ""
    # Clean LaTeX formatting from title
    title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
    title = re.sub(r"\\[a-zA-Z]+", "", title)
    title = re.sub(r"[{}]", "", title)
    title = " ".join(title.split())

    author = _extract_braced(tex, "author") or ""
    # Clean author field
    author = re.sub(r"\\footnotemark\[[^\]]*\]", "", author)
    author = re.sub(r"\\footnotetext\[[^\]]*\]\{[^}]*\}", "", author)
    author = re.sub(r"\\and\b", ", ", author)
    author = re.sub(r"\\\\\s*", " ", author)
    author = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", author)
    author = re.sub(r"\\[a-zA-Z]+", "", author)
    author = re.sub(r"[{}$^]", "", author)
    author = re.sub(r"\s+", " ", author).strip()

    # Try to extract affiliations from author block
    affiliations = ""
    aff_patterns = [
        r"\\(?:affiliation|institute|institution)\{([^}]+)\}",
        r"\\(?:small|footnotesize|normalsize)\s*\$\^?\{?\d+\}?\$\s*([^\\$]+)",
    ]
    affs = []
    for pat in aff_patterns:
        affs.extend(re.findall(pat, tex))
    if affs:
        affiliations = "; ".join(a.strip() for a in affs)
    else:
        # Try to extract from author block text after backslash-backslash
        lines = author.split(",")
        aff_lines = [l.strip() for l in lines if any(kw in l.lower() for kw in ["university", "institute", "school", "lab", "department"])]
        if aff_lines:
            affiliations = "; ".join(aff_lines)
            # Remove affiliation lines from author
            author_parts = [l.strip() for l in lines if l.strip() not in aff_lines]
            author = ", ".join(p for p in author_parts if p)

    return {
        "poster_title": title,
        "authors": author,
        "affiliations": affiliations,
    }


def extract_abstract(tex: str) -> str:
    """Extract abstract text."""
    m = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL
    )
    if m:
        abstract = m.group(1).strip()
        abstract = _clean_latex(abstract)
        return abstract
    return ""


def _clean_latex(text: str) -> str:
    """Clean LaTeX markup from text, preserving readability."""
    # Remove \cite{...}
    text = re.sub(r"\\cite[pt]?\{[^}]*\}", "", text)
    # Remove \ref{...} and \label{...}
    text = re.sub(r"\\(?:ref|label|eqref)\{[^}]*\}", "", text)
    # Convert \textbf{...} -> bold text
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\underline\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\text(?:tt|sf|sc|rm)\{([^}]*)\}", r"\1", text)
    # Remove \footnote{...}
    text = re.sub(r"\\footnote\{[^}]*\}", "", text)
    # Simple math: $x$ -> x
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    # Remove remaining LaTeX commands but keep argument
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # Remove lone commands
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Clean up braces, tildes
    text = re.sub(r"[{}~]", " ", text)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_sections(tex: str) -> List[dict]:
    """
    Split document body into sections.

    Returns:
        [
            {"title": "Introduction", "content": "...", "subsections": [...]},
            ...
        ]
    """
    # Find document body
    body_m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex, re.DOTALL)
    body = body_m.group(1) if body_m else tex

    # Split by \section{...}
    section_pattern = re.compile(r"\\section\*?\{([^}]+)\}")
    splits = list(section_pattern.finditer(body))

    if not splits:
        return [{"title": "Content", "content": _clean_latex(body)}]

    sections = []
    for i, m in enumerate(splits):
        title = m.group(1).strip()
        title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
        title = re.sub(r"\\[a-zA-Z]+", "", title)
        title = re.sub(r"[{}]", "", title).strip()

        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(body)
        content = body[start:end].strip()

        # Extract subsections
        subsections = _extract_subsections(content)

        # Clean the main content
        clean_content = _clean_latex(content)

        sections.append({
            "title": title,
            "content": clean_content,
            "subsections": subsections,
        })

    return sections


def _extract_subsections(tex: str) -> List[dict]:
    """Extract subsections from within a section."""
    pattern = re.compile(r"\\subsection\*?\{([^}]+)\}")
    splits = list(pattern.finditer(tex))
    if not splits:
        return []

    subsections = []
    for i, m in enumerate(splits):
        title = m.group(1).strip()
        title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
        title = re.sub(r"[{}]", "", title).strip()

        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(tex)
        content = _clean_latex(tex[start:end].strip())

        subsections.append({"title": title, "content": content})

    return subsections


def extract_figures(tex: str) -> List[dict]:
    """
    Extract figure environments with captions and includegraphics paths.

    Returns:
        [
            {
                "figure_id": 1,
                "caption": "A description...",
                "graphics_paths": ["fig/example.pdf"],
                "label": "fig:example",
            },
            ...
        ]
    """
    figures = []
    pattern = re.compile(
        r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.DOTALL
    )
    for i, m in enumerate(pattern.finditer(tex)):
        block = m.group(1)

        # Extract caption
        caption = _extract_braced(block, "caption") or ""
        caption = _clean_latex(caption)

        # Extract \includegraphics paths
        graphics = re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block
        )

        # Extract label
        label = ""
        label_m = re.search(r"\\label\{([^}]+)\}", block)
        if label_m:
            label = label_m.group(1)

        figures.append({
            "figure_id": i + 1,
            "caption": caption,
            "graphics_paths": graphics,
            "label": label,
        })

    return figures


def extract_tables(tex: str) -> List[dict]:
    """
    Extract table environments with captions.

    Returns:
        [
            {
                "table_id": 1,
                "caption": "Results on benchmark...",
                "label": "tab:results",
            },
            ...
        ]
    """
    tables = []
    pattern = re.compile(
        r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.DOTALL
    )
    for i, m in enumerate(pattern.finditer(tex)):
        block = m.group(1)

        caption = _extract_braced(block, "caption") or ""
        caption = _clean_latex(caption)

        label = ""
        label_m = re.search(r"\\label\{([^}]+)\}", block)
        if label_m:
            label = label_m.group(1)

        tabular_m = re.search(
            r"\\begin\{tabular[*x]?\}(\{[^}]*\})?(.*?)\\end\{tabular[*x]?\}",
            block, re.DOTALL,
        )
        content = ""
        if tabular_m:
            col_spec = (tabular_m.group(1) or "").strip()
            content = f"\\begin{{tabular}}{col_spec}\n{tabular_m.group(2).strip()}\n\\end{{tabular}}"

        tables.append({
            "table_id": i + 1,
            "caption": caption,
            "label": label,
            "content": content,
        })

    return tables


def find_main_tex(extracted_dir: str) -> str:
    """
    Find the main .tex file in the extracted directory.
    Heuristic: look for \\documentclass or \\begin{document}.
    """
    tex_files = list(Path(extracted_dir).rglob("*.tex"))

    # First pass: look for \documentclass
    for tf in tex_files:
        content = _read_tex(str(tf))
        if r"\documentclass" in content:
            return str(tf)

    # Fallback: look for common names
    for name in ("main.tex", "paper.tex", "camera-ready.tex", "article.tex"):
        candidate = os.path.join(extracted_dir, name)
        if os.path.exists(candidate):
            return candidate

    # Last resort: first .tex file
    if tex_files:
        return str(tex_files[0])

    raise FileNotFoundError(f"No .tex file found in {extracted_dir}")


def parse_latex_project(extracted_dir: str) -> dict:
    """
    Parse a complete LaTeX project directory into structured JSON.

    Args:
        extracted_dir: Path to extracted zip contents

    Returns:
        {
            "meta": {"poster_title": ..., "authors": ..., "affiliations": ...},
            "abstract": "...",
            "sections": [...],
            "figures": [...],
            "tables": [...],
            "main_tex_path": "...",
        }
    """
    print("  Finding main .tex file...")
    main_tex_path = find_main_tex(extracted_dir)
    print(f"  Main file: {main_tex_path}")

    print("  Reading and resolving includes...")
    raw_tex = _read_tex(main_tex_path)
    base_dir = os.path.dirname(main_tex_path)
    full_tex = _resolve_inputs(raw_tex, base_dir)
    full_tex = _strip_comments(full_tex)

    print("  Extracting metadata...")
    meta = extract_metadata(full_tex)

    print("  Extracting abstract...")
    abstract = extract_abstract(full_tex)

    print("  Extracting sections...")
    sections = extract_sections(full_tex)

    print("  Extracting figures...")
    figures = extract_figures(full_tex)

    print("  Extracting tables...")
    tables = extract_tables(full_tex)

    print(f"  Parsed: {len(sections)} sections, {len(figures)} figures, {len(tables)} tables")

    return {
        "meta": meta,
        "abstract": abstract,
        "sections": sections,
        "figures": figures,
        "tables": tables,
        "main_tex_path": main_tex_path,
    }
