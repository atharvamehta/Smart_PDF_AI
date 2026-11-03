from pathlib import Path
from typing import Union

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ..services import document_service as documents, pdf_service as pdf
from ..services.errors import ToolError

router = APIRouter(prefix="/pdf", tags=["pdf tools"])

Pages = Union[str, list[int]]


def _row(doc_id: int):
    row = documents.get_row(doc_id)
    if not row:
        raise HTTPException(404, f"Document {doc_id} not found.")
    return row


def _finish(src_row, out: Path, name: str, op: str, detail: str, background: BackgroundTasks) -> dict:
    """Register the produced PDF as a normal document so it can be previewed, chatted with, merged again."""
    doc = documents.register_pdf(out, name, source=op)
    documents.record_operation(src_row["id"] if src_row else None, op, detail, name)
    background.add_task(documents.index_in_background, doc["id"])
    return doc


def _stem(row) -> str:
    return Path(row["filename"]).stem


class MergeReq(BaseModel):
    document_ids: list[int] = Field(min_length=2)
    name: str | None = None


class PagesReq(BaseModel):
    document_id: int
    pages: Pages


class ReorderReq(BaseModel):
    document_id: int
    order: list[int]


class RotateReq(BaseModel):
    document_id: int
    pages: Pages
    degrees: int = 90


class EditReq(BaseModel):
    document_id: int
    ops: list[dict]


@router.post("/merge")
def merge(req: MergeReq, background: BackgroundTasks):
    rows = [_row(i) for i in req.document_ids]
    out = pdf.merge([documents.stored_path(r) for r in rows])
    name = (req.name or "merged").removesuffix(".pdf") + ".pdf"
    return _finish(None, out, name, "merge", ",".join(map(str, req.document_ids)), background)


@router.post("/split")
def split(req: PagesReq, background: BackgroundTasks):
    row = _row(req.document_id)
    out = pdf.extract(documents.stored_path(row), req.pages)
    label = req.pages if isinstance(req.pages, str) else ",".join(map(str, req.pages))
    return _finish(row, out, f"{_stem(row)} (pages {label.replace(' ', '')}).pdf", "extract", label, background)


@router.post("/delete-pages")
def delete_pages(req: PagesReq, background: BackgroundTasks):
    row = _row(req.document_id)
    out = pdf.delete_pages(documents.stored_path(row), req.pages)
    return _finish(row, out, f"{_stem(row)} (edited).pdf", "delete-pages", str(req.pages), background)


@router.post("/reorder")
def reorder(req: ReorderReq, background: BackgroundTasks):
    row = _row(req.document_id)
    out = pdf.reorder(documents.stored_path(row), req.order)
    return _finish(row, out, f"{_stem(row)} (reordered).pdf", "reorder", ",".join(map(str, req.order)), background)


@router.post("/rotate")
def rotate(req: RotateReq, background: BackgroundTasks):
    row = _row(req.document_id)
    out = pdf.rotate(documents.stored_path(row), req.pages, req.degrees)
    return _finish(row, out, f"{_stem(row)} (rotated).pdf", "rotate", f"{req.pages}@{req.degrees}", background)


@router.post("/edit")
def edit(req: EditReq, background: BackgroundTasks):
    row = _row(req.document_id)
    try:
        out = pdf.annotate(documents.stored_path(row), req.ops)
    except (KeyError, ValueError, TypeError) as exc:
        raise ToolError("One of the annotations was malformed.") from exc
    return _finish(row, out, f"{_stem(row)} (annotated).pdf", "annotate", f"{len(req.ops)} ops", background)
