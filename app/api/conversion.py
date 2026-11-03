from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services import conversion_service as conversion, document_service as documents

router = APIRouter(tags=["convert"])


class ConvertReq(BaseModel):
    document_id: int
    target: Literal["docx", "pptx"]


@router.post("/convert")
def convert(req: ConvertReq):
    """PDF -> DOCX / PPTX. (DOCX / PPTX -> PDF happens automatically on upload.)"""
    row = documents.get_row(req.document_id)
    if not row:
        raise HTTPException(404, "Document not found.")
    src = documents.stored_path(row)
    fn = conversion.pdf_to_docx if req.target == "docx" else conversion.pdf_to_pptx
    out = fn(src, row["filename"])
    documents.record_operation(row["id"], f"pdf-to-{req.target}", row["filename"], out.name)
    return {"filename": out.name.split("_", 1)[-1], "download_url": f"/api/outputs/{out.name}"}
