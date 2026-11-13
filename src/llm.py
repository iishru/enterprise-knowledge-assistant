# llm.py - picks the chat model: Groq (free hosted API) or Ollama (free, runs on your laptop).
import os

from langchain_core.language_models.chat_models import BaseChatModel

import config


class MissingAPIKeyError(RuntimeError):
    """Raised when the Groq provider is chosen but GROQ_API_KEY is not set."""


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the chosen provider (default: LLM_PROVIDER in config)."""
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com, "
                "put it in your .env file as GROQ_API_KEY=..., or switch LLM_PROVIDER to ollama."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.GROQ_MODEL, temperature=config.TEMPERATURE, api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Needs the Ollama app running locally and the model pulled: `ollama pull llama3.2`
        return ChatOllama(model=config.OLLAMA_MODEL, temperature=config.TEMPERATURE)

    raise ValueError(f"Unknown LLM provider '{provider}'. Use 'groq' or 'ollama'.")
