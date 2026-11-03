"""Thin SQLite helper. No ORM: the schema is small and plain SQL is faster to read."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    stored_name   TEXT NOT NULL,
    file_type     TEXT NOT NULL DEFAULT 'pdf',
    size          INTEGER NOT NULL DEFAULT 0,
    page_count    INTEGER NOT NULL DEFAULT 0,
    text_chars    INTEGER NOT NULL DEFAULT 0,
    scanned_pages INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'indexing',   -- indexing | ready | error
    error         TEXT,
    source        TEXT NOT NULL DEFAULT 'upload',
    summary       TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number  INTEGER NOT NULL,
    text         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(document_id, page_number);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number  INTEGER NOT NULL,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, id);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,           -- user | assistant
    content         TEXT NOT NULL,
    meta            TEXT,                    -- JSON: sources, grounded, provider
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS operations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id    INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    operation_type TEXT NOT NULL,
    input          TEXT,
    output         TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    config.ensure_dirs()
    with connect() as conn:
        conn.executescript(SCHEMA)


def fetch_all(sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(sql, params).fetchone()


def execute(sql: str, params: tuple | list = ()) -> int:
    with connect() as conn:
        return conn.execute(sql, params).lastrowid or 0


def execute_many(sql: str, rows: list[tuple[Any, ...]]) -> None:
    with connect() as conn:
        conn.executemany(sql, rows)
