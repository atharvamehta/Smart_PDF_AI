from fastapi import APIRouter
from pydantic import BaseModel

from ..core import config
from ..services import embedding_service, llm_service as llm, summary_service
from ..services.errors import ToolError
from .sse import event, stream_response

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/status")
def status():
    return {
        "providers": llm.available_providers(),
        "models": llm.provider_models(),
        "embeddings": embedding_service.status(),
        "chunk_words": config.CHUNK_WORDS,
        "top_k": config.TOP_K,
    }


@router.post("/test")
async def test():
    return await llm.test_providers()


class SummaryReq(BaseModel):
    document_id: int
    refresh: bool = False


@router.post("/summary")
async def summary(req: SummaryReq):
    async def gen():
        try:
            async for ev in summary_service.stream_summary(req.document_id, req.refresh):
                yield event(ev)
        except (ToolError, llm.LLMError) as exc:
            yield event({"type": "error", "message": str(exc)})

    return stream_response(gen())
