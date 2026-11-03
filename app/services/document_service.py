"""Document registry: turning a file on disk into a row, pages, and a search index."""
from __future__ import annotations

import logging
from pathlib import Path

from ..core import config, database as db
from . import extraction_service as extract, rag_service

log = logging.getLogger(__name__)


def to_dict(row) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "size": row["size"],
        "page_count": row["page_count"],
        "text_chars": row["text_chars"],
        "scanned_pages": row["scanned_pages"],
        "status": row["status"],
        "error": row["error"],
        "source": row["source"],
        "has_summary": bool(row["summary"]),
        "created_at": row["created_at"],
    }


def get_row(doc_id: int):
    return db.fetch_one("SELECT * FROM documents WHERE id=?", (doc_id,))


def list_documents() -> list[dict]:
    return [to_dict(r) for r in db.fetch_all("SELECT * FROM documents ORDER BY id DESC")]


def register_pdf(path: Path, filename: str, source: str = "upload") -> dict:
    """Read a stored PDF, save its page text, and return the new document.
    Text extraction is synchronous and fast; embedding is left for `index_in_background`."""
    pages = extract.extract_pages(path)
    chars, scanned = extract.summarize_extraction(pages)
    doc_id = db.execute(
        """INSERT INTO documents (filename, stored_name, file_type, size, page_count,
                                  text_chars, scanned_pages, status, source)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (filename, path.name, "pdf", path.stat().st_size, len(pages), chars, scanned,
         "indexing" if chars else "ready", source),
    )
    db.execute_many(
        "INSERT INTO pages (document_id, page_number, text) VALUES (?,?,?)",
        [(doc_id, i, t) for i, t in enumerate(pages, 1)],
    )
    return to_dict(get_row(doc_id))


def index_in_background(doc_id: int) -> None:
    """Runs in a worker thread after the upload response is already sent."""
    row = get_row(doc_id)
    if not row or row["status"] == "ready":
        return  # nothing to index (image-only PDF) or already done
    try:
        rag_service.index_document(doc_id)
        db.execute("UPDATE documents SET status='ready', error=NULL WHERE id=?", (doc_id,))
    except Exception as exc:  # keep the library usable even if embedding fails
        log.exception("Indexing failed for document %s", doc_id)
        db.execute("UPDATE documents SET status='error', error=? WHERE id=?", (str(exc)[:300], doc_id))


def delete_document(doc_id: int) -> bool:
    row = get_row(doc_id)
    if not row:
        return False
    db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    rag_service.drop_document(doc_id)
    extract.clear_thumbs(doc_id)
    (config.UPLOAD_DIR / row["stored_name"]).unlink(missing_ok=True)
    return True


def record_operation(doc_id: int | None, kind: str, inp: str, out: str) -> None:
    db.execute(
        "INSERT INTO operations (document_id, operation_type, input, output) VALUES (?,?,?,?)",
        (doc_id, kind, inp, out),
    )


def stored_path(row) -> Path:
    return config.UPLOAD_DIR / row["stored_name"]
