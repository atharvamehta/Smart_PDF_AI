"""Local embeddings. FastEmbed (ONNX, no PyTorch) is fast to load and light on RAM.
If the model cannot be downloaded, a deterministic hashing embedder keeps the app
working with keyword-style retrieval instead of crashing."""
from __future__ import annotations

import logging
import re
import threading
import time
import zlib

import numpy as np

from ..core import config

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (m / norms).astype("float32")


class HashEmbedder:
    """Signed feature hashing of unigrams + bigrams. No download, no model."""

    name = "hash384"
    dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            words = _TOKEN.findall(text.lower())
            feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for f in feats:
                h = zlib.crc32(f.encode())
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=str(config.MODEL_CACHE_DIR))
        self.name = model_name.split("/")[-1]
        self.dim = len(next(iter(self.model.embed(["warm up"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.embed(texts, batch_size=64)), dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        try:
            return _normalize(np.array(list(self.model.query_embed(text)), dtype="float32"))
        except Exception:
            return self.embed([text])


_lock = threading.Lock()
_hash = HashEmbedder()
_model: FastEmbedder | None = None
_failed = False
_loader: threading.Thread | None = None
_deadline = 0.0
CACHED_LOAD_WAIT = 3.0  # a cached model loads in ~1-2 s; a download takes much longer


def _load_model() -> None:
    global _model, _failed
    try:
        model = FastEmbedder(config.EMBEDDING_MODEL)
        with _lock:
            _model = model
        log.info("Embedding model ready: %s", model.name)
    except Exception as exc:
        with _lock:
            _failed = True
        log.warning("Embedding model unavailable (%s). Using keyword-style fallback.", str(exc)[:120])


def start_loading() -> None:
    """Begin loading the model in the background (idempotent)."""
    global _loader, _deadline
    with _lock:
        if _loader is None and config.EMBEDDING_BACKEND != "hash":
            _deadline = time.monotonic() + CACHED_LOAD_WAIT
            _loader = threading.Thread(target=_load_model, daemon=True)
            _loader.start()


def get_embedder() -> HashEmbedder | FastEmbedder:
    """Never blocks for long. Returns the real model once loaded, otherwise the hashing
    fallback. Indexes are tagged by embedder, so documents are re-embedded automatically
    the first time the real model is used on them."""
    if _model is not None:
        return _model
    if config.EMBEDDING_BACKEND == "hash" or _failed:
        return _hash
    start_loading()
    if _loader is not None:
        # One shared grace period measured from when loading began, never per call.
        _loader.join(timeout=max(0.0, _deadline - time.monotonic()))
    return _model or _hash


def status() -> dict:
    if _model is not None:
        state, backend = "ready", _model.name
    elif config.EMBEDDING_BACKEND == "hash" or _failed:
        state, backend = "fallback", _hash.name
    else:
        state, backend = "loading", _hash.name
    return {"state": state, "backend": backend}
