"""
Figure discovery, PDF-to-PNG conversion, and base64 encoding utilities.
"""

import os
import base64
import glob
from pathlib import Path
from typing import Dict, Optional

import fitz  # pymupdf
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg"}
VECTOR_EXTENSIONS = {".pdf", ".eps"}


def discover_figures(root_dir: str) -> Dict[str, str]:
    """
    Recursively find all image and vector figure files under root_dir.

    Returns:
        {relative_path: absolute_path}
    """
    figures = {}
    root = Path(root_dir)
    for ext in IMAGE_EXTENSIONS | VECTOR_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            rel = str(p.relative_to(root))
            figures[rel] = str(p.resolve())
    return figures


def pdf_to_png(pdf_path: str, output_path: Optional[str] = None, dpi: int = 200) -> str:
    """
    Convert a single-page PDF figure to PNG using pymupdf.

    Args:
        pdf_path: Path to the PDF file
        output_path: Where to save the PNG (default: same name with .png)
        dpi: Resolution for rendering

    Returns:
        Path to the generated PNG file
    """
    if output_path is None:
        output_path = str(Path(pdf_path).with_suffix(".png"))

    doc = fitz.open(pdf_path)
    page = doc[0]  # first page
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    doc.close()

    return output_path


def convert_all_vector_figures(root_dir: str, dpi: int = 200) -> Dict[str, str]:
    """
    Convert all PDF/EPS figures under root_dir to PNG.

    Returns:
        {original_relative_path: png_absolute_path}
    """
    conversions = {}
    root = Path(root_dir)
    for ext in VECTOR_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            png_path = str(p.with_suffix(".png"))
            if not os.path.exists(png_path):
                try:
                    pdf_to_png(str(p), png_path, dpi=dpi)
                except Exception as e:
                    print(f"  Warning: failed to convert {p}: {e}")
                    continue
            rel = str(p.relative_to(root))
            conversions[rel] = png_path
    return conversions


def get_image_dimensions(img_path: str) -> tuple:
    """Return (width, height) of an image file."""
    with Image.open(img_path) as img:
        return img.size


def image_to_base64(img_path: str, max_width: int = 1200) -> str:
    """
    Read an image, optionally resize, and return a base64 data URI.

    Args:
        img_path: Path to the image file
        max_width: Maximum width in pixels (proportional resize if exceeded)

    Returns:
        Data URI string: "data:image/png;base64,..."
    """
    with Image.open(img_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            fmt = "PNG"
            mime = "image/png"
        else:
            img = img.convert("RGB")
            fmt = "JPEG"
            mime = "image/jpeg"

        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        import io
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return f"data:{mime};base64,{b64}"


def build_figure_registry(
    root_dir: str,
    max_width: int = 1200,
    dpi: int = 200,
) -> Dict[str, dict]:
    """
    Build a complete figure registry for the extracted LaTeX project.

    Steps:
    1. Convert all PDF/EPS figures to PNG
    2. Discover all raster images
    3. Build base64 data URIs and collect metadata

    Returns:
        {
            "fig/example.pdf": {
                "png_path": "/abs/path/fig/example.png",
                "width": 800,
                "height": 600,
                "base64_uri": "data:image/png;base64,...",
            },
            "images/photo.jpg": {
                "png_path": "/abs/path/images/photo.jpg",
                "width": 1024,
                "height": 768,
                "base64_uri": "data:image/jpeg;base64,...",
            }
        }
    """
    print("  Converting vector figures to PNG...")
    conversions = convert_all_vector_figures(root_dir, dpi=dpi)

    print("  Discovering all figures...")
    all_figures = discover_figures(root_dir)

    registry = {}
    root = Path(root_dir)

    for rel_path, abs_path in all_figures.items():
        ext = Path(rel_path).suffix.lower()

        # For vector formats, use the converted PNG
        if ext in VECTOR_EXTENSIONS:
            png_path = conversions.get(rel_path)
            if png_path is None or not os.path.exists(png_path):
                continue
        else:
            png_path = abs_path

        try:
            w, h = get_image_dimensions(png_path)
            b64 = image_to_base64(png_path, max_width=max_width)
        except Exception as e:
            print(f"  Warning: skipping {rel_path}: {e}")
            continue

        # Also register without extension for LaTeX \includegraphics{fig/name}
        registry[rel_path] = {
            "png_path": png_path,
            "width": w,
            "height": h,
            "base64_uri": b64,
        }

        # Register path without extension (LaTeX often omits it)
        no_ext = str(Path(rel_path).with_suffix(""))
        if no_ext not in registry:
            registry[no_ext] = registry[rel_path]

        # Also register with .png extension for converted PDFs
        if ext in VECTOR_EXTENSIONS:
            png_rel = str(Path(rel_path).with_suffix(".png"))
            if png_rel not in registry:
                registry[png_rel] = registry[rel_path]

    print(f"  Figure registry: {len(registry)} entries")
    return registry
