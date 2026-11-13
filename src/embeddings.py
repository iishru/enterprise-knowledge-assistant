# embeddings.py - builds the embedding model that turns text into vectors.
# Real model: all-MiniLM-L6-v2 (free, runs on CPU, downloaded from Hugging Face on first use).
# "fake" backend: random-but-repeatable vectors for offline tests only (they have no meaning).
from functools import lru_cache

from langchain_core.embeddings import Embeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the embedding model. Built once and reused."""
    if config.EMBEDDING_BACKEND == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=config.EMBEDDING_DIM)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # unit-length vectors -> cosine similarity
    )
