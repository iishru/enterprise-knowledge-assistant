# test_app.py - runs the Streamlit app headlessly (streamlit's AppTest) with a fake LLM
# and word-overlap test embeddings, in a temporary database.
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

import config
from src.ingest import ingest_path
from src.llm import MissingAPIKeyError
from tests.helpers import KeywordEmbeddings

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture()
def env(monkeypatch, tmp_path):
    st.cache_resource.clear()
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "db"))
    monkeypatch.setattr("src.vectorstore.get_embeddings", lambda: KeywordEmbeddings())
    yield
    st.cache_resource.clear()


def test_app_starts_with_empty_knowledge_base(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.endswith("Enterprise Knowledge Assistant")
    assert any("Nothing indexed yet" in i.value for i in at.sidebar.info)


def test_empty_knowledge_base_message(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hello?").run()
    assert not at.exception
    assert "knowledge base is empty" in at.chat_message[-1].markdown[0].value


def test_question_gets_answer_and_sources(env, monkeypatch):
    ingest_path(SAMPLE_DIR)
    monkeypatch.setattr("src.llm.get_llm", lambda provider=None: FakeListChatModel(
        responses=["Sick leave is 10 days per year. [leave_policy.pdf, p.1]"]))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("How many sick leave days do I get?").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "10 days" in at.chat_message[1].markdown[0].value
    assert any(e.label.startswith("Sources") for e in at.expander)


def test_missing_api_key_shows_friendly_message(env, monkeypatch):
    ingest_path(SAMPLE_DIR)

    def no_key(provider=None):
        raise MissingAPIKeyError("GROQ_API_KEY is not set.")

    monkeypatch.setattr("src.llm.get_llm", no_key)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("How many sick leave days do I get?").run()
    assert not at.exception
    assert "GROQ_API_KEY is not set" in at.chat_message[-1].markdown[0].value


def test_clear_chat_button_empties_conversation(env, monkeypatch):
    ingest_path(SAMPLE_DIR)
    monkeypatch.setattr("src.llm.get_llm", lambda provider=None: FakeListChatModel(responses=["ok"]))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("sick leave days").run()
    assert len(at.chat_message) == 2
    next(b for b in at.sidebar.button if b.label == "Clear chat").click().run()
    assert len(at.chat_message) == 0
