import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core import database as db
from ..services import llm_service as llm, prompts, rag_service
from .sse import event, stream_response

router = APIRouter(prefix="/chat", tags=["chat"])

HISTORY_MESSAGES = 6
HISTORY_CHARS = 1200


class ChatReq(BaseModel):
    question: str
    document_ids: list[int] = []
    conversation_id: int | None = None
    allow_general: bool = True


def _history(conv_id: int) -> list[dict]:
    rows = db.fetch_all(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?", (conv_id, HISTORY_MESSAGES)
    )
    return [{"role": r["role"], "content": r["content"][:HISTORY_CHARS]} for r in reversed(rows)]


def _search_query(question: str, history: list[dict]) -> str:
    """Short follow-ups ('explain the second one') borrow the previous question for retrieval."""
    if len(question.split()) < 8:
        prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{prev} {question}".strip()
    return question


def _not_ready(doc_ids: list[int]) -> str | None:
    rows = db.fetch_all(
        f"SELECT filename, status, text_chars FROM documents WHERE id IN ({','.join('?' * len(doc_ids))})", doc_ids
    )
    busy = [r["filename"] for r in rows if r["status"] == "indexing"]
    if busy:
        return f"'{busy[0]}' is still being prepared. Try again in a few seconds."
    if rows and all(r["text_chars"] == 0 for r in rows):
        return "These PDFs have no selectable text (they look scanned), so I can't read them yet."
    return None


@router.post("")
async def chat(req: ChatReq):
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "Type a question first.")

    conv_id = req.conversation_id
    if conv_id and not db.fetch_one("SELECT id FROM conversations WHERE id=?", (conv_id,)):
        conv_id = None
    if not conv_id:
        conv_id = db.execute("INSERT INTO conversations (title) VALUES (?)", (question[:60],))
    history = _history(conv_id)
    db.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?,?,?)", (conv_id, "user", question))

    async def gen():
        yield event({"type": "init", "conversation_id": conv_id})
        try:
            hits = []
            if req.document_ids:
                if problem := _not_ready(req.document_ids):
                    yield event({"type": "error", "message": problem})
                    return
                hits = await asyncio.to_thread(
                    rag_service.retrieve, _search_query(question, history), req.document_ids
                )

            if req.document_ids:
                fallback = prompts.FALLBACK_ALLOWED if req.allow_general else prompts.FALLBACK_DENIED
                system = prompts.CHAT_SYSTEM.replace("{fallback}", fallback)
                user = f"Context passages:\n\n{prompts.build_context(hits)}\n\nQuestion: {question}"
            else:
                system, user = prompts.NO_DOCS_SYSTEM, question

            messages = history + [{"role": "user", "content": user}]
            focus = _search_query(question, history)
            sources = [
                {"n": i, "doc_id": h.doc_id, "filename": h.filename, "page": h.page,
                 "snippet": rag_service.best_passage(h.text, focus), "score": round(h.score, 3)}
                for i, h in enumerate(hits, 1)
            ]

            answer, buffer, resolved, grounded, provider = [], "", False, True, ""
            marker = prompts.NOT_IN_DOC
            t0 = time.perf_counter()

            def start_event() -> str:
                return event({"type": "start", "grounded": grounded, "provider": provider,
                              "sources": sources if grounded and req.document_ids else [],
                              "first_token_ms": round((time.perf_counter() - t0) * 1000)})

            async for chunk in llm.stream(system, messages):
                provider, text = chunk["provider"], chunk["text"]
                if not resolved:
                    buffer += text
                    probe = buffer.lstrip()
                    if len(probe) < len(marker) and marker.startswith(probe):
                        continue  # could still turn into the not-in-document marker
                    resolved = True
                    if probe.startswith(marker):
                        grounded, text = False, probe[len(marker):].lstrip()
                    else:
                        text = buffer
                    yield start_event()
                    if not text:
                        continue
                answer.append(text)
                yield event({"type": "token", "text": text})

            if not resolved:  # very short reply that never left the buffer
                text = buffer
                if buffer.lstrip().startswith(marker):
                    grounded, text = False, buffer.lstrip()[len(marker):].lstrip()
                yield start_event()
                answer.append(text)
                yield event({"type": "token", "text": text})

            final = "".join(answer).strip()
            meta = {"sources": sources if grounded and req.document_ids else [],
                    "grounded": grounded, "provider": provider, "has_docs": bool(req.document_ids)}
            db.execute(
                "INSERT INTO messages (conversation_id, role, content, meta) VALUES (?,?,?,?)",
                (conv_id, "assistant", final, json.dumps(meta)),
            )
            yield event({"type": "done", "grounded": grounded, "provider": provider})
        except llm.LLMError as exc:
            yield event({"type": "error", "message": str(exc)})

    return stream_response(gen())


@router.get("")
def list_conversations():
    rows = db.fetch_all("SELECT id, title, created_at FROM conversations ORDER BY id DESC LIMIT 50")
    return [dict(r) for r in rows]


@router.get("/{conv_id}")
def get_conversation(conv_id: int):
    conv = db.fetch_one("SELECT id, title FROM conversations WHERE id=?", (conv_id,))
    if not conv:
        raise HTTPException(404, "Conversation not found.")
    rows = db.fetch_all("SELECT role, content, meta FROM messages WHERE conversation_id=? ORDER BY id", (conv_id,))
    return {
        "id": conv["id"],
        "title": conv["title"],
        "messages": [
            {"role": r["role"], "content": r["content"], "meta": json.loads(r["meta"]) if r["meta"] else None}
            for r in rows
        ],
    }


@router.delete("/{conv_id}")
def delete_conversation(conv_id: int):
    db.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    return {"ok": True}
