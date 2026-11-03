"""Central configuration. Everything tunable lives here and comes from .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


# ── Paths ────────────────────────────────────────────────────────────────
BACKEND_DIR = ROOT / "backend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "vector_index"
THUMB_DIR = DATA_DIR / "thumbs"
MODEL_CACHE_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "app.db"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# ── Upload rules ─────────────────────────────────────────────────────────
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 50)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx"}

# ── LLM providers (tried in this order, first working one wins) ──────────
LLM_PROVIDERS = [p.strip() for p in _env("LLM_PROVIDERS", "gemini,groq").lower().split(",") if p.strip()]

GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.5-flash-lite")

GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-120b")

OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "openrouter/free")

OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2:3b")

# ── Embeddings & retrieval ───────────────────────────────────────────────
EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "auto").lower()  # auto | fastembed | hash
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_WORDS = _int("CHUNK_WORDS", 300)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 50)
TOP_K = _int("TOP_K", 5)

# ── Summary ──────────────────────────────────────────────────────────────
SINGLE_PASS_WORDS = _int("SINGLE_PASS_WORDS", 5000)
SECTION_WORDS = _int("SECTION_WORDS", 3000)


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, OUTPUT_DIR, DATA_DIR, INDEX_DIR, THUMB_DIR, MODEL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
