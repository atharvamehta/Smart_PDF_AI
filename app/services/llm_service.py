"""Provider-agnostic streaming LLM client with automatic fallback.

Order comes from LLM_PROVIDERS in .env (default: gemini, then groq). If a provider
fails *before* producing any text (rate limit, bad key, network), the next one is used,
so a demo does not die on a single 429."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

import httpx

from ..core import config

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


# ── Shared HTTP client (keeps TLS connections warm = faster first token) ──
_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def _http() -> httpx.AsyncClient:
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=8.0))
        _client_loop = loop
    return _client


async def _check(name: str, r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    body = (await r.aread()).decode(errors="ignore")[:300]
    if r.status_code == 429:
        raise LLMError(f"{name}: rate limit reached, try again shortly")
    if r.status_code in (401, 403):
        raise LLMError(f"{name}: API key rejected")
    raise LLMError(f"{name}: HTTP {r.status_code} {body}")


# ── Providers ────────────────────────────────────────────────────────────
async def _gemini(system: str, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:streamGenerateContent?alt=sse"
    body: dict = {
        "contents": [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    async with _http().stream("POST", url, json=body, headers={"x-goog-api-key": config.GEMINI_API_KEY}) as r:
        await _check("gemini", r)
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            for cand in json.loads(payload).get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("thought"):
                        continue  # skip hidden reasoning parts
                    if part.get("text"):
                        yield part["text"]


async def _openai_compat(
    name: str, base_url: str, key: str, model: str,
    system: str, messages: list[dict], max_tokens: int, temperature: float,
) -> AsyncIterator[str]:
    body: dict = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if name == "groq" and "gpt-oss" in model:
        body["reasoning_effort"] = "low"  # much faster first token
    async with _http().stream(
        "POST", f"{base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    ) as r:
        await _check(name, r)
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            for choice in json.loads(payload).get("choices", []):
                text = (choice.get("delta") or {}).get("content")
                if text:
                    yield text


async def _ollama(system: str, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[str]:
    body = {
        "model": config.OLLAMA_MODEL,
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    async with _http().stream("POST", f"{config.OLLAMA_HOST}/api/chat", json=body) as r:
        await _check("ollama", r)
        async for line in r.aiter_lines():
            if line.strip():
                text = json.loads(line).get("message", {}).get("content")
                if text:
                    yield text


def _streamer(name: str, system, messages, max_tokens, temperature) -> AsyncIterator[str]:
    if name == "gemini":
        return _gemini(system, messages, max_tokens, temperature)
    if name == "groq":
        return _openai_compat("groq", "https://api.groq.com/openai/v1", config.GROQ_API_KEY,
                              config.GROQ_MODEL, system, messages, max_tokens, temperature)
    if name == "openrouter":
        return _openai_compat("openrouter", "https://openrouter.ai/api/v1", config.OPENROUTER_API_KEY,
                              config.OPENROUTER_MODEL, system, messages, max_tokens, temperature)
    if name == "ollama":
        return _ollama(system, messages, max_tokens, temperature)
    raise LLMError(f"Unknown provider '{name}'")


# ── Public API ───────────────────────────────────────────────────────────
def available_providers() -> list[str]:
    keys = {"gemini": config.GEMINI_API_KEY, "groq": config.GROQ_API_KEY, "openrouter": config.OPENROUTER_API_KEY}
    return [p for p in config.LLM_PROVIDERS if p == "ollama" or keys.get(p)]


def provider_models() -> dict[str, str]:
    return {"gemini": config.GEMINI_MODEL, "groq": config.GROQ_MODEL,
            "openrouter": config.OPENROUTER_MODEL, "ollama": config.OLLAMA_MODEL}


async def stream(
    system: str, messages: list[dict], max_tokens: int = 1200, temperature: float = 0.2,
) -> AsyncIterator[dict]:
    """Yield {"provider": str, "text": str}. Falls through providers until one starts answering."""
    providers = available_providers()
    if not providers:
        raise LLMError("No AI provider configured. Add GEMINI_API_KEY (free) to your .env file.")
    errors: list[str] = []
    for name in providers:
        started = False
        try:
            async for piece in _streamer(name, system, messages, max_tokens, temperature):
                started = True
                yield {"provider": name, "text": piece}
            if started:
                return
            errors.append(f"{name}: empty reply")
        except (LLMError, httpx.HTTPError) as exc:
            if started:
                raise LLMError(f"{name} stopped mid-answer: {exc}") from exc
            log.warning("Provider %s failed: %s", name, exc)
            errors.append(str(exc) or f"{name}: connection failed")
    raise LLMError(" | ".join(errors))


async def complete(system: str, messages: list[dict], max_tokens: int = 700, temperature: float = 0.2) -> str:
    return "".join([c["text"] async for c in stream(system, messages, max_tokens, temperature)])


async def test_providers() -> list[dict]:
    """Time-to-first-token for each configured provider (Settings > Test connection)."""
    results = []
    for name in available_providers():
        t0 = time.perf_counter()
        try:
            async for _ in _streamer(name, "", [{"role": "user", "content": "Reply with the single word: ok"}], 16, 0):
                break
            results.append({"provider": name, "model": provider_models()[name], "ok": True,
                            "ms": round((time.perf_counter() - t0) * 1000)})
        except Exception as exc:
            results.append({"provider": name, "model": provider_models()[name], "ok": False, "error": str(exc)[:200]})
    return results
