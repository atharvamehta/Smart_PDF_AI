"""End-to-end API tests. LLM calls are faked, embeddings use the offline hash backend."""
import json
import os
import sys
from pathlib import Path

os.environ["EMBEDDING_BACKEND"] = "hash"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core import config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for name, sub in [("UPLOAD_DIR", "uploads"), ("OUTPUT_DIR", "outputs"), ("DATA_DIR", "data"),
                      ("INDEX_DIR", "data/index"), ("THUMB_DIR", "data/thumbs"), ("MODEL_CACHE_DIR", "data/models")]:
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "app.db")
    from app.main import app
    with TestClient(app) as c:
        yield c


def make_pdf(tmp_path, pages: list[str], name="sample.pdf") -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    path = tmp_path / name
    doc.save(path)
    return path


def upload(client, path: Path, mime="application/pdf"):
    with path.open("rb") as f:
        return client.post("/api/documents/upload", files=[("files", (path.name, f, mime))]).json()


NLP = ["Natural language processing studies how computers understand human language. Tokenization splits text into words.",
       "Embeddings map words to vectors. Retrieval augmented generation grounds answers in documents.",
       "Transformers use attention. Applications include translation, summarization and chatbots."]


def test_upload_and_render(client, tmp_path):
    res = upload(client, make_pdf(tmp_path, NLP))
    assert res["documents"][0]["status"] == "indexing"  # upload replies before embedding finishes
    doc = client.get(f"/api/documents/{res['documents'][0]['id']}").json()  # the UI polls like this
    assert doc["page_count"] == 3 and doc["status"] == "ready" and doc["text_chars"] > 100
    img = client.get(f"/api/documents/{doc['id']}/pages/1/image?w=200")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/documents/{doc['id']}/file").status_code == 200


def test_rejects_bad_files(client, tmp_path):
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    assert "not supported" in upload(client, bad, "application/octet-stream")["errors"][0]["error"]
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not really a pdf")
    assert "could not be read" in upload(client, fake)["errors"][0]["error"]


def test_page_tools(client, tmp_path):
    d = upload(client, make_pdf(tmp_path, NLP))["documents"][0]
    i = d["id"]
    ex = client.post("/api/pdf/split", json={"document_id": i, "pages": "1-2"}).json()
    assert ex["page_count"] == 2
    dl = client.post("/api/pdf/delete-pages", json={"document_id": i, "pages": [2]}).json()
    assert dl["page_count"] == 2
    ro = client.post("/api/pdf/reorder", json={"document_id": i, "order": [3, 1, 2]}).json()
    assert ro["page_count"] == 3
    first = client.get(f"/api/documents/{ro['id']}/text?page=1").json()["text"]
    assert "Transformers" in first
    mg = client.post("/api/pdf/merge", json={"document_ids": [i, ex["id"]], "name": "both"}).json()
    assert mg["page_count"] == 5 and mg["filename"] == "both.pdf"
    rt = client.post("/api/pdf/rotate", json={"document_id": i, "pages": "1", "degrees": 90}).json()
    assert rt["page_count"] == 3


def test_tool_errors_are_readable(client, tmp_path):
    i = upload(client, make_pdf(tmp_path, NLP))["documents"][0]["id"]
    r = client.post("/api/pdf/split", json={"document_id": i, "pages": "9"})
    assert r.status_code == 422 and "between 1 and 3" in r.json()["detail"]
    r = client.post("/api/pdf/delete-pages", json={"document_id": i, "pages": "1-3"})
    assert "every page" in r.json()["detail"]
    r = client.post("/api/pdf/reorder", json={"document_id": i, "order": [1, 2]})
    assert r.status_code == 422


def test_annotate(client, tmp_path):
    i = upload(client, make_pdf(tmp_path, NLP))["documents"][0]["id"]
    ops = [{"type": "text", "page": 1, "x": .1, "y": .1, "text": "Reviewed", "color": "#c0392b"},
           {"type": "highlight", "page": 1, "x": .1, "y": .3, "w": .5, "h": .05, "color": "#ffe066"},
           {"type": "rect", "page": 2, "x": .2, "y": .2, "w": .3, "h": .1},
           {"type": "draw", "page": 2, "points": [[.1, .5], [.3, .6], [.5, .5]]}]
    out = client.post("/api/pdf/edit", json={"document_id": i, "ops": ops}).json()
    assert "Reviewed" in client.get(f"/api/documents/{out['id']}/text?page=1").json()["text"]


def test_convert_and_office_upload(client, tmp_path):
    i = upload(client, make_pdf(tmp_path, NLP))["documents"][0]["id"]
    for target in ("docx", "pptx"):
        r = client.post("/api/convert", json={"document_id": i, "target": target}).json()
        assert r["filename"].endswith(target)
        assert client.get(r["download_url"]).status_code == 200
    # round trip: the generated DOCX uploads back and becomes a searchable PDF
    docx = client.get(client.post("/api/convert", json={"document_id": i, "target": "docx"}).json()["download_url"]).content
    p = tmp_path / "report.docx"
    p.write_bytes(docx)
    back = upload(client, p, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert back["documents"][0]["filename"] == "report.pdf" and back["documents"][0]["page_count"] >= 1


def sse(resp) -> list[dict]:
    return [json.loads(l[5:]) for l in resp.text.splitlines() if l.startswith("data:")]


def fake_llm(monkeypatch, pieces):
    from app.services import llm_service

    async def fake_stream(system, messages, max_tokens=1200, temperature=0.2):
        for p in pieces:
            yield {"provider": "fake", "text": p}

    monkeypatch.setattr(llm_service, "stream", fake_stream)


def test_chat_grounded_with_sources(client, tmp_path, monkeypatch):
    i = upload(client, make_pdf(tmp_path, NLP))["documents"][0]["id"]
    fake_llm(monkeypatch, ["RAG grounds answers ", "in documents [1]."])
    ev = sse(client.post("/api/chat", json={"question": "What does retrieval augmented generation do?", "document_ids": [i]}))
    types = [e["type"] for e in ev]
    assert types[0] == "init" and "start" in types and types[-1] == "done"
    start = next(e for e in ev if e["type"] == "start")
    assert start["grounded"] and start["sources"][0]["page"] == 2
    cid = ev[0]["conversation_id"]
    hist = client.get(f"/api/chat/{cid}").json()
    assert [m["role"] for m in hist["messages"]] == ["user", "assistant"]
    assert hist["messages"][1]["meta"]["sources"]


def test_chat_not_in_document_marker(client, tmp_path, monkeypatch):
    i = upload(client, make_pdf(tmp_path, NLP))["documents"][0]["id"]
    fake_llm(monkeypatch, ["[NOT_IN", "_DOCUMENT] Paris is the capital of France."])
    ev = sse(client.post("/api/chat", json={"question": "Capital of France?", "document_ids": [i]}))
    start = next(e for e in ev if e["type"] == "start")
    assert start["grounded"] is False and start["sources"] == []
    text = "".join(e["text"] for e in ev if e["type"] == "token")
    assert text.startswith("Paris") and "NOT_IN" not in text


def test_multi_doc_and_summary(client, tmp_path, monkeypatch):
    a = upload(client, make_pdf(tmp_path, NLP, "a.pdf"))["documents"][0]["id"]
    b = upload(client, make_pdf(tmp_path, ["Photosynthesis converts sunlight into chemical energy in plants."], "b.pdf"))["documents"][0]["id"]
    fake_llm(monkeypatch, ["ok [1][2]"])
    ev = sse(client.post("/api/chat", json={"question": "Compare language and photosynthesis topics", "document_ids": [a, b]}))
    docs = {s["doc_id"] for s in next(e for e in ev if e["type"] == "start")["sources"]}
    assert docs == {a, b}
    fake_llm(monkeypatch, ["## Overview\nA summary."])
    ev = sse(client.post("/api/ai/summary", json={"document_id": a}))
    assert ev[-1]["type"] == "done" and not ev[-1]["cached"]
    ev = sse(client.post("/api/ai/summary", json={"document_id": a}))
    assert ev[-1]["cached"] is True


def test_not_ready_and_delete(client, tmp_path):
    i = upload(client, make_pdf(tmp_path, NLP))["documents"][0]["id"]
    assert client.delete(f"/api/documents/{i}").json()["ok"]
    assert client.get(f"/api/documents/{i}").status_code == 404


def test_locate_highlights_wrapped_passage(client, tmp_path):
    text = ("Retrieval augmented generation grounds language model answers in external documents. "
            "Instead of relying on memorised facts, the system fetches relevant passages first, "
            "then asks the model to answer using only that evidence.")
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(60, 60, 330, 400), text, fontsize=11)
    p = tmp_path / "wrap.pdf"
    doc.save(p)
    i = upload(client, p)["documents"][0]["id"]
    r = client.get(f"/api/documents/{i}/pages/1/locate", params={"q": "system fetches relevant passages first, then asks the model"}).json()
    assert r["width"] > 0 and len(r["rects"]) >= 2  # wrapped over two lines
    assert all(0 <= x["x"] <= 1 and 0 <= x["y"] <= 1 for x in r["rects"])
    assert client.get(f"/api/documents/{i}/pages/1/locate").json()["rects"] == []  # size only


def test_best_passage_prefers_matching_window():
    from app.services.rag_service import best_passage
    filler = "lorem ipsum dolor sit amet " * 12
    text = filler + "photosynthesis converts sunlight into chemical energy inside chloroplasts " + filler
    out = best_passage(text, "How does photosynthesis convert sunlight?")
    assert "photosynthesis" in out and len(out.split()) == 35
