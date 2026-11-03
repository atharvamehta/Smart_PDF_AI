import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..core import config, database as db
from ..services import conversion_service as conversion, document_service as documents, extraction_service as extract
from ..services.errors import ToolError

router = APIRouter(tags=["documents"])


async def _save_upload(f: UploadFile, suffix: str) -> Path:
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    stored = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size, too_big = 0, False
    with stored.open("wb") as out:
        while chunk := await f.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                too_big = True
                break
            out.write(chunk)
    if too_big:
        stored.unlink(missing_ok=True)
        raise ToolError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if size == 0:
        stored.unlink(missing_ok=True)
        raise ToolError("The file is empty.")
    return stored


async def _ingest(f: UploadFile, background: BackgroundTasks) -> dict:
    name = Path(f.filename or "file").name
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ToolError(f"'{ext or name}' is not supported. Upload a PDF, DOCX or PPTX file.")
    stored = await _save_upload(f, ext)
    source = "upload"
    try:
        if ext != ".pdf":  # Word / PowerPoint are converted once, so every feature works on PDFs
            pdf_path = await run_in_threadpool(conversion.office_to_pdf, stored)
            stored.unlink(missing_ok=True)
            stored, name, source = pdf_path, Path(name).stem + ".pdf", f"converted from {ext[1:].upper()}"
        doc = await run_in_threadpool(documents.register_pdf, stored, name, source)
    except Exception:
        stored.unlink(missing_ok=True)
        raise
    background.add_task(documents.index_in_background, doc["id"])
    return doc


@router.post("/documents/upload")
async def upload(background: BackgroundTasks, files: list[UploadFile] = File(...)):
    docs, errors = [], []
    for f in files:
        try:
            docs.append(await _ingest(f, background))
        except ToolError as exc:
            errors.append({"filename": f.filename, "error": str(exc)})
    return {"documents": docs, "errors": errors}


@router.get("/documents")
def list_documents():
    return documents.list_documents()


def _row_or_404(doc_id: int):
    row = documents.get_row(doc_id)
    if not row:
        raise HTTPException(404, "Document not found.")
    return row


@router.get("/documents/{doc_id}")
def get_document(doc_id: int):
    return documents.to_dict(_row_or_404(doc_id))


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    _row_or_404(doc_id)
    documents.delete_document(doc_id)
    return {"ok": True}


@router.get("/documents/{doc_id}/file")
def get_file(doc_id: int, download: bool = False):
    row = _row_or_404(doc_id)
    return FileResponse(
        documents.stored_path(row),
        media_type="application/pdf",
        filename=row["filename"],
        content_disposition_type="attachment" if download else "inline",
    )


@router.get("/documents/{doc_id}/pages")
def page_info(doc_id: int):
    _row_or_404(doc_id)
    rows = db.fetch_all(
        "SELECT page_number, length(trim(text)) AS chars FROM pages WHERE document_id=? ORDER BY page_number", (doc_id,)
    )
    return [{"page": r["page_number"], "chars": r["chars"]} for r in rows]


@router.get("/documents/{doc_id}/pages/{page}/image")
def page_image(doc_id: int, page: int, w: int = 300):
    row = _row_or_404(doc_id)
    path = extract.render_page(doc_id, row["stored_name"], page, w)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/documents/{doc_id}/pages/{page}/locate")
def locate(doc_id: int, page: int, q: str = ""):
    row = _row_or_404(doc_id)
    return extract.locate_text(row["stored_name"], page, q)


@router.get("/documents/{doc_id}/text")
def page_text(doc_id: int, page: int):
    _row_or_404(doc_id)
    row = db.fetch_one("SELECT text FROM pages WHERE document_id=? AND page_number=?", (doc_id, page))
    if not row:
        raise HTTPException(404, "Page not found.")
    return {"page": page, "text": row["text"]}


@router.get("/outputs/{name}")
def download_output(name: str):
    path = config.OUTPUT_DIR / Path(name).name  # strip any directory parts
    if not path.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(path, filename=path.name.split("_", 1)[-1])
