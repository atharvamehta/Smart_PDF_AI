"""Page-level PDF operations. Every function writes a NEW file and returns its path."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import pymupdf

from ..core import config
from .errors import ToolError
from .extraction_service import open_pdf


def _new_path() -> Path:
    return config.UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"


def _save(doc: pymupdf.Document, out: Path) -> Path:
    doc.save(out, garbage=3, deflate=True)
    doc.close()
    return out


def parse_pages(spec: str | list[int], page_count: int) -> list[int]:
    """'1-3, 5, 8-' or [1,2,5] -> zero-based page indexes (order preserved)."""
    if isinstance(spec, list):
        numbers = spec
    else:
        numbers = []
        for part in re.split(r"[,\s]+", spec.strip()):
            if not part:
                continue
            m = re.fullmatch(r"(\d*)-(\d*)", part)
            if m:
                a, b = int(m[1] or 1), int(m[2] or page_count)
                if a > b:
                    raise ToolError(f"'{part}' is a backwards range.")
                numbers.extend(range(a, b + 1))
            elif part.isdigit():
                numbers.append(int(part))
            else:
                raise ToolError(f"Can't understand '{part}'. Use something like 1-3, 5, 8-")
    if not numbers:
        raise ToolError("Select at least one page.")
    bad = [n for n in numbers if n < 1 or n > page_count]
    if bad:
        raise ToolError(f"Pages must be between 1 and {page_count} (got {bad[0]}).")
    return [n - 1 for n in numbers]


def merge(paths: list[Path]) -> Path:
    if len(paths) < 2:
        raise ToolError("Pick at least two PDFs to merge.")
    merged = pymupdf.open()
    for p in paths:
        with open_pdf(p) as src:
            merged.insert_pdf(src)
    return _save(merged, _new_path())


def extract(path: Path, pages: str | list[int]) -> Path:
    doc = open_pdf(path)
    doc.select(parse_pages(pages, doc.page_count))
    return _save(doc, _new_path())


def delete_pages(path: Path, pages: str | list[int]) -> Path:
    doc = open_pdf(path)
    remove = set(parse_pages(pages, doc.page_count))
    keep = [i for i in range(doc.page_count) if i not in remove]
    if not keep:
        doc.close()
        raise ToolError("You can't delete every page.")
    doc.select(keep)
    return _save(doc, _new_path())


def reorder(path: Path, order: list[int]) -> Path:
    doc = open_pdf(path)
    if sorted(order) != list(range(1, doc.page_count + 1)):
        doc.close()
        raise ToolError("The new order must list every page exactly once.")
    doc.select([n - 1 for n in order])
    return _save(doc, _new_path())


def rotate(path: Path, pages: str | list[int], degrees: int) -> Path:
    if degrees % 90:
        raise ToolError("Rotation must be a multiple of 90 degrees.")
    doc = open_pdf(path)
    for i in parse_pages(pages, doc.page_count):
        page = doc[i]
        page.set_rotation((page.rotation + degrees) % 360)
    return _save(doc, _new_path())


# ── Overlay editing ──────────────────────────────────────────────────────
def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0.1, 0.1, 0.1)
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def annotate(path: Path, ops: list[dict]) -> Path:
    """Apply overlay operations. Coordinates are fractions (0-1) of the page size,
    so the browser preview and the PDF never disagree about scale."""
    if not ops:
        raise ToolError("Nothing to save: add at least one annotation first.")
    doc = open_pdf(path)
    for op in ops:
        n = int(op.get("page", 1))
        if not 1 <= n <= doc.page_count:
            continue
        page = doc[n - 1]
        w, h = page.rect.width, page.rect.height
        kind = op.get("type")
        color = _rgb(op.get("color", "#222222"))

        if kind == "text":
            size = float(op.get("size", 14))
            x, y = float(op["x"]) * w, float(op["y"]) * h
            for i, line in enumerate(str(op.get("text", "")).split("\n")):
                page.insert_text((x, y + size + i * size * 1.25), line, fontsize=size, color=color, fontname="helv")
        elif kind in ("highlight", "rect"):
            rect = pymupdf.Rect(
                float(op["x"]) * w,
                float(op["y"]) * h,
                (float(op["x"]) + float(op["w"])) * w,
                (float(op["y"]) + float(op["h"])) * h,
            )
            if kind == "highlight":
                page.draw_rect(rect, color=None, fill=color, fill_opacity=0.35, overlay=True)
            else:
                page.draw_rect(rect, color=color, width=float(op.get("width", 1.5)), overlay=True)
        elif kind == "draw":
            pts = [pymupdf.Point(float(px) * w, float(py) * h) for px, py in op.get("points", [])]
            if len(pts) >= 2:
                page.draw_polyline(pts, color=color, width=float(op.get("width", 2)), overlay=True)
    return _save(doc, _new_path())
