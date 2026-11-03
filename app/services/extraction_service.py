"""Fast text extraction and page rendering (PyMuPDF only, no OCR, no model calls)."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from ..core import config
from .errors import ToolError

# A page with fewer characters than this is treated as image-only (scanned).
SCANNED_PAGE_THRESHOLD = 20


def open_pdf(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # corrupt / not a pdf
        raise ToolError("This file could not be read as a PDF.") from exc
    if doc.needs_pass:
        doc.close()
        raise ToolError("This PDF is password protected. Remove the password and upload again.")
    return doc


def extract_pages(path: Path) -> list[str]:
    """Return the text of every page, about 1 ms per page.
    (Not using sort=True: it re-orders blocks geometrically and is ~14x slower.)"""
    with open_pdf(path) as doc:
        return [page.get_text("text") for page in doc]


def summarize_extraction(pages: list[str]) -> tuple[int, int]:
    """(total characters, number of image-only pages)."""
    chars = sum(len(p.strip()) for p in pages)
    scanned = sum(1 for p in pages if len(p.strip()) < SCANNED_PAGE_THRESHOLD)
    return chars, scanned


def render_page(doc_id: int, stored_name: str, page_number: int, width: int) -> Path:
    """Render one page to a cached JPEG and return its path."""
    width = max(80, min(width, 2000))
    out = config.THUMB_DIR / f"{doc_id}_{page_number}_{width}.jpg"
    if out.exists():
        return out
    with open_pdf(config.UPLOAD_DIR / stored_name) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise ToolError(f"Page {page_number} does not exist.")
        page = doc[page_number - 1]
        scale = width / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        out.write_bytes(pix.tobytes("jpg", jpg_quality=82))
    return out


def locate_text(stored_name: str, page_number: int, query: str = "") -> dict:
    """Page size in points plus normalised (0-1) rectangles where `query` appears.
    Used to draw the yellow 'this is where the answer came from' highlight."""
    with open_pdf(config.UPLOAD_DIR / stored_name) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise ToolError(f"Page {page_number} does not exist.")
        page = doc[page_number - 1]
        w, h = page.rect.width, page.rect.height
        q = re.sub(r"\s+", " ", query).strip()
        rects = page.search_for(q) if q else []
        if q and not rects:  # hyphenation or odd spacing: fall back to short sliding phrases
            words = q.split()
            for i in range(0, max(1, len(words) - 3), 3):
                rects += page.search_for(" ".join(words[i : i + 4]))
    return {
        "width": w,
        "height": h,
        "rects": [{"x": r.x0 / w, "y": r.y0 / h, "w": (r.x1 - r.x0) / w, "h": (r.y1 - r.y0) / h} for r in rects],
    }


def clear_thumbs(doc_id: int) -> None:
    for f in config.THUMB_DIR.glob(f"{doc_id}_*.jpg"):
        f.unlink(missing_ok=True)
