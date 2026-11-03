"""PDF <-> DOCX / PPTX conversion using free local tools."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pymupdf

from ..core import config
from .errors import ToolError
from .extraction_service import open_pdf


def find_soffice() -> str | None:
    """Locate LibreOffice. Windows/macOS installers do NOT add it to PATH, so besides
    PATH we check the standard install folders. LIBREOFFICE_PATH in .env overrides all."""
    override = os.getenv("LIBREOFFICE_PATH", "").strip().strip('"')
    if override and Path(override).is_file():
        return override
    for name in ("soffice", "libreoffice", "soffice.exe"):
        found = shutil.which(name)
        if found:
            return found

    candidates: list[Path] = []
    if sys.platform.startswith("win"):
        bases = [os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)"), r"C:\Program Files", r"C:\Program Files (x86)"]
        candidates = [Path(b) / "LibreOffice" / "program" / "soffice.exe" for b in bases if b]
    elif sys.platform == "darwin":
        candidates = [Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")]
    return next((str(c) for c in candidates if c.is_file()), None)


def _soffice() -> str:
    exe = find_soffice()
    if not exe:
        raise ToolError(
            "LibreOffice was not found. Install it from libreoffice.org, then restart the server. "
            "If it is installed in a custom folder, add LIBREOFFICE_PATH=C:\\path\\to\\soffice.exe to your .env file."
        )
    return exe


def office_to_pdf(src: Path) -> Path:
    """DOCX / PPTX -> PDF through headless LibreOffice."""
    exe = _soffice()
    with tempfile.TemporaryDirectory() as tmp:
        # Separate profile dir lets conversions run even if LibreOffice is open on the desktop.
        # as_uri() builds a valid file URL on every OS (file:///C:/... on Windows).
        profile = (Path(tmp) / "profile").as_uri()
        cmd = [exe, f"-env:UserInstallation={profile}", "--headless",
               "--convert-to", "pdf", "--outdir", tmp, str(src)]
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console flash on Windows
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError("Conversion took too long and was stopped.") from exc
        except subprocess.CalledProcessError as exc:
            raise ToolError("LibreOffice could not convert this file.") from exc
        produced = next(Path(tmp).glob("*.pdf"), None)
        if not produced:
            raise ToolError("LibreOffice finished but produced no PDF.")
        out = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
        shutil.move(produced, out)
        return out


def pdf_to_docx(src: Path, display_name: str) -> Path:
    from pdf2docx import Converter

    out = config.OUTPUT_DIR / f"{uuid.uuid4().hex[:8]}_{Path(display_name).stem}.docx"
    cv = Converter(str(src))
    try:
        cv.convert(str(out))
    except Exception as exc:
        raise ToolError("This PDF could not be converted to Word (scanned or heavily designed pages are hard).") from exc
    finally:
        cv.close()
    return out


def pdf_to_pptx(src: Path, display_name: str) -> Path:
    """One slide per page: the page image as background, extracted text kept in notes."""
    from pptx import Presentation
    from pptx.util import Emu

    out = config.OUTPUT_DIR / f"{uuid.uuid4().hex[:8]}_{Path(display_name).stem}.pptx"
    prs = Presentation()
    blank = prs.slide_layouts[6]
    with open_pdf(src) as doc, tempfile.TemporaryDirectory() as tmp:
        first = doc[0].rect
        prs.slide_width = Emu(9144000 * 1.0 * 1.0)
        prs.slide_height = Emu(int(9144000 * first.height / first.width))
        for i, page in enumerate(doc):
            img = Path(tmp) / f"p{i}.jpg"
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            img.write_bytes(pix.tobytes("jpg", jpg_quality=85))
            slide = prs.slides.add_slide(blank)
            slide.shapes.add_picture(str(img), 0, 0, width=prs.slide_width, height=prs.slide_height)
            slide.notes_slide.notes_text_frame.text = page.get_text("text").strip()
    prs.save(out)
    return out