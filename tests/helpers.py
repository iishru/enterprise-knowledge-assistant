# helpers.py - test-only embeddings that really measure word overlap (unlike the random "fake"
# backend), so retrieval order and the relevance threshold can be tested without downloads.
import re

from langchain_core.embeddings import Embeddings

VOCAB = [
    "leave", "casual", "sick", "earned", "days", "password", "vpn", "security",
    "hotel", "travel", "claim", "salary", "payroll", "laptop", "ticket", "gift",
    "review", "rating", "home", "office", "remote",
]


class KeywordEmbeddings(Embeddings):
    """Vector = counts of known words (normalised). Texts with no known word look 'unrelated'."""

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        vec = [float(words.count(w)) for w in VOCAB]
        vec.append(0.01)  # tiny constant so a text with no known words is never an all-zero vector
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
