"""Chunking, FAISS indexing and retrieval. One FAISS index per document, so that
deleting a file or selecting a subset for multi-PDF chat is trivial."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import faiss
import numpy as np

from ..core import config, database as db
from .embedding_service import get_embedder


@dataclass
class Hit:
    doc_id: int
    filename: str
    page: int
    text: str
    score: float


@dataclass
class _Entry:
    index: faiss.Index
    chunks: list[tuple[int, str]]  # (page_number, text) in index order


_cache: dict[tuple[int, str], _Entry] = {}
_cache_lock = threading.Lock()


# ── Chunking ─────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # un-hyphenate line breaks
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Sliding word window inside each page so every chunk keeps an exact page number."""
    step = max(1, config.CHUNK_WORDS - config.CHUNK_OVERLAP)
    out: list[tuple[int, int, str]] = []
    for page_no, text in pages:
        words = _clean(text).split()
        for start in range(0, len(words), step):
            if start > 0 and len(words) - start <= config.CHUNK_OVERLAP:
                break  # tail already covered by the previous chunk's overlap
            out.append((page_no, len(out), " ".join(words[start : start + config.CHUNK_WORDS])))
    return out


_STOP = frozenset("the a an of to in is are was were and or for on with as by at from that this it be what how why "
                  "which who does do explain tell me about give".split())


def best_passage(text: str, query: str, size: int = 35) -> str:
    """The `size`-word window of a chunk that shares the most terms with the question.
    It is what we show and highlight as the evidence, instead of the chunk's opening lines."""
    words = text.split()
    if len(words) <= size:
        return text
    terms = {w for w in re.findall(r"\w+", query.lower()) if w not in _STOP and len(w) > 2}
    lowered = [re.sub(r"\W+", "", w.lower()) for w in words]
    best_start, best_score = 0, -1
    for start in range(0, len(words) - size + 1, 3):
        score = sum(1 for w in lowered[start : start + size] if w in terms)
        if score > best_score:
            best_start, best_score = start, score
    return " ".join(words[best_start : best_start + size])


# ── Indexing ─────────────────────────────────────────────────────────────
def _index_path(doc_id: int, tag: str):
    return config.INDEX_DIR / f"{doc_id}-{tag}.faiss"


def _build_faiss(doc_id: int) -> _Entry:
    """(Re)build the FAISS index for a document from its chunk rows."""
    emb = get_embedder()
    rows = db.fetch_all("SELECT page_number, content FROM chunks WHERE document_id=? ORDER BY id", (doc_id,))
    index = faiss.IndexFlatIP(emb.dim)
    texts = [r["content"] for r in rows]
    for i in range(0, len(texts), 64):
        index.add(emb.embed(texts[i : i + 64]))
    for old in config.INDEX_DIR.glob(f"{doc_id}-*.faiss"):
        old.unlink(missing_ok=True)
    if rows:
        faiss.write_index(index, str(_index_path(doc_id, emb.name)))
    entry = _Entry(index, [(r["page_number"], r["content"]) for r in rows])
    with _cache_lock:
        _cache[(doc_id, emb.name)] = entry
    return entry


def index_document(doc_id: int) -> None:
    """Chunk the stored page text and build the vector index."""
    pages = db.fetch_all("SELECT page_number, text FROM pages WHERE document_id=? ORDER BY page_number", (doc_id,))
    chunks = chunk_pages([(r["page_number"], r["text"]) for r in pages])
    with db.connect() as conn:
        conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        conn.executemany(
            "INSERT INTO chunks (document_id, page_number, chunk_index, content) VALUES (?,?,?,?)",
            [(doc_id, p, i, t) for p, i, t in chunks],
        )
    _build_faiss(doc_id)


def drop_document(doc_id: int) -> None:
    with _cache_lock:
        for key in [k for k in _cache if k[0] == doc_id]:
            del _cache[key]
    for f in config.INDEX_DIR.glob(f"{doc_id}-*.faiss"):
        f.unlink(missing_ok=True)


def _load(doc_id: int) -> _Entry:
    tag = get_embedder().name
    with _cache_lock:
        entry = _cache.get((doc_id, tag))
    if entry:
        return entry
    path = _index_path(doc_id, tag)
    if path.exists():
        rows = db.fetch_all("SELECT page_number, content FROM chunks WHERE document_id=? ORDER BY id", (doc_id,))
        entry = _Entry(faiss.read_index(str(path)), [(r["page_number"], r["content"]) for r in rows])
        with _cache_lock:
            _cache[(doc_id, tag)] = entry
        return entry
    return _build_faiss(doc_id)  # embedder changed since indexing: re-embed once


# ── Retrieval ────────────────────────────────────────────────────────────
def retrieve(query: str, doc_ids: list[int], k: int | None = None) -> list[Hit]:
    k = k or config.TOP_K
    if not doc_ids:
        return []
    names = {
        r["id"]: r["filename"]
        for r in db.fetch_all(f"SELECT id, filename FROM documents WHERE id IN ({','.join('?' * len(doc_ids))})", doc_ids)
    }
    qv = get_embedder().embed_query(query).reshape(1, -1)

    per_doc: list[list[Hit]] = []
    for did in doc_ids:
        if did not in names:
            continue
        entry = _load(did)
        if entry.index.ntotal == 0:
            continue
        scores, ids = entry.index.search(qv, min(k, entry.index.ntotal))
        per_doc.append(
            [
                Hit(did, names[did], entry.chunks[i][0], entry.chunks[i][1], float(s))
                for s, i in zip(scores[0], ids[0])
                if i >= 0
            ]
        )

    # Guarantee each selected document is represented, then fill by score.
    chosen: list[Hit] = [hits[0] for hits in per_doc if hits] if len(per_doc) > 1 else []
    seen = {(h.doc_id, h.page, h.text[:40]) for h in chosen}
    rest = sorted((h for hits in per_doc for h in hits), key=lambda h: -h.score)
    for h in rest:
        if len(chosen) >= k:
            break
        if (h.doc_id, h.page, h.text[:40]) not in seen:
            chosen.append(h)
    return sorted(chosen, key=lambda h: -h.score)[:k]
