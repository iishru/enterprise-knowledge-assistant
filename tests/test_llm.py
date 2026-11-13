# test_llm.py - provider selection and the clear error for a missing Groq key. No network needed.
import pytest

from src.llm import MissingAPIKeyError, get_llm


def test_missing_groq_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
        get_llm("groq")


def test_groq_model_is_built_when_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-test")
    llm = get_llm("groq")
    assert llm.model_name == "llama-3.1-8b-instant"
    assert llm.temperature < 1e-6  # ChatGroq turns 0 into 1e-8 (Groq rejects exactly 0)


def test_ollama_model_is_built_without_connecting():
    llm = get_llm("ollama")
    assert llm.model == "llama3.2"


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        get_llm("openai")
