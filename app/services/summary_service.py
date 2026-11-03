"""Document summarisation. Short docs: one streamed call. Long docs: parallel
map step over sections, then one streamed reduce step."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..core import config, database as db
from . import llm_service as llm, prompts
from .errors import ToolError

MAX_SECTIONS = 10


def _page_texts(doc_id: int) -> list[tuple[int, str]]:
    rows = db.fetch_all("SELECT page_number, text FROM pages WHERE document_id=? ORDER BY page_number", (doc_id,))
    return [(r["page_number"], r["text"]) for r in rows if r["text"].strip()]


def _sections(pages: list[tuple[int, str]], total_words: int) -> list[str]:
    """Group pages into roughly equal sections, at most MAX_SECTIONS of them."""
    target = max(config.SECTION_WORDS, total_words // MAX_SECTIONS + 1)
    sections, buf, count = [], [], 0
    for n, text in pages:
        words = len(text.split())
        if buf and count + words > target:
            sections.append("\n".join(buf))
            buf, count = [], 0
        buf.append(f"[Page {n}]\n{text.strip()}")
        count += words
    if buf:
        sections.append("\n".join(buf))
    return sections


async def stream_summary(doc_id: int, refresh: bool = False) -> AsyncIterator[dict]:
    row = db.fetch_one("SELECT filename, summary FROM documents WHERE id=?", (doc_id,))
    if not row:
        raise ToolError("Document not found.")
    if row["summary"] and not refresh:
        yield {"type": "token", "text": row["summary"]}
        yield {"type": "done", "cached": True}
        return

    pages = _page_texts(doc_id)
    if not pages:
        raise ToolError("This PDF has no selectable text (it looks scanned), so there is nothing to summarise yet.")
    total_words = sum(len(t.split()) for _, t in pages)

    if total_words <= config.SINGLE_PASS_WORDS:
        body = "\n".join(f"[Page {n}]\n{t.strip()}" for n, t in pages)
        user = f"Document: {row['filename']}\n\n{body}"
    else:
        sections = _sections(pages, total_words)
        yield {"type": "progress", "message": f"Reading {len(sections)} sections…"}
        gate = asyncio.Semaphore(3)

        async def note(i: int, text: str) -> str:
            async with gate:
                return await llm.complete(prompts.MAP_SYSTEM, [{"role": "user", "content": text}], max_tokens=500)

        notes = await asyncio.gather(*(note(i, s) for i, s in enumerate(sections)))
        joined = "\n\n".join(f"Section {i}:\n{n}" for i, n in enumerate(notes, 1))
        user = f"{prompts.REDUCE_NOTE}\nDocument: {row['filename']}\n\n{joined}"

    parts: list[str] = []
    provider = ""
    async for chunk in llm.stream(prompts.SUMMARY_SYSTEM, [{"role": "user", "content": user}], max_tokens=1200):
        provider = chunk["provider"]
        parts.append(chunk["text"])
        yield {"type": "token", "text": chunk["text"]}
    db.execute("UPDATE documents SET summary=? WHERE id=?", ("".join(parts), doc_id))
    yield {"type": "done", "provider": provider, "cached": False}
