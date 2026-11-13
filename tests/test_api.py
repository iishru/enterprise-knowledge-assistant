# test_api.py - tests the REST API with FastAPI's TestClient, a fake LLM and word-overlap embeddings.
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import api
import config
from src.llm import MissingAPIKeyError
from tests.helpers import KeywordEmbeddings

LEAVE_NOTE = b"# Leave\n\nSick leave is ten days per year. Casual leave is twelve days per year."


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "db"))
    monkeypatch.setattr("src.vectorstore.get_embeddings", lambda: KeywordEmbeddings())
    return TestClient(api.app)


def upload_note(client: TestClient):
    return client.post("/ingest", files=[("files", ("leave_note.md", LEAVE_NOTE, "text/markdown"))])


def test_health_on_empty_knowledge_base(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "chunks": 0, "documents": []}


def test_ask_on_empty_knowledge_base_is_409(client):
    response = client.post("/ask", json={"question": "sick leave days?"})
    assert response.status_code == 409


def test_ingest_then_ask_returns_answer_with_sources(client, monkeypatch):
    summary = upload_note(client).json()
    assert summary["files"] == 1 and summary["chunks_added"] >= 1
    assert client.get("/health").json()["documents"] == ["leave_note.md"]

    monkeypatch.setattr(api, "get_llm", lambda: FakeListChatModel(responses=["10 days. [leave_note.md]"]))
    response = client.post("/ask", json={"question": "How many sick leave days?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "10 days. [leave_note.md]"
    assert body["sources"][0]["source"] == "leave_note.md"
    assert body["latency_ms"] >= 0


def test_follow_up_history_is_accepted(client, monkeypatch):
    upload_note(client)
    llm = FakeListChatModel(responses=["How many sick leave days?", "10 days. [leave_note.md]"])
    monkeypatch.setattr(api, "get_llm", lambda: llm)
    response = client.post("/ask", json={
        "question": "And the other type?",
        "history": [{"question": "How many casual leave days?", "answer": "12 days."}],
    })
    assert response.status_code == 200
    assert response.json()["standalone_question"] == "How many sick leave days?"


def test_missing_api_key_is_503(client, monkeypatch):
    upload_note(client)

    def no_key():
        raise MissingAPIKeyError("GROQ_API_KEY is not set.")

    monkeypatch.setattr(api, "get_llm", no_key)
    response = client.post("/ask", json={"question": "How many sick leave days?"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_unsupported_upload_is_400(client):
    response = client.post("/ingest", files=[("files", ("photo.png", b"x", "image/png"))])
    assert response.status_code == 400


def test_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
