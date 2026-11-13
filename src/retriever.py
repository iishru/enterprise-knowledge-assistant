# retriever.py - finds the chunks most related to a question.
# Returns (Document, score) pairs; score is between 0 and 1 (higher = more similar).
# If even the best chunk is below the relevance threshold we return [] = "nothing relevant".
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.vectorstore import get_vectorstore


def retrieve(
    question: str,
    k: int | None = None,
    search_type: str | None = None,
    threshold: float | None = None,
    vectorstore: Chroma | None = None,
) -> list[tuple[Document, float]]:
    """Return up to k relevant chunks with scores, or [] if nothing is relevant enough."""
    k = config.TOP_K if k is None else k
    search_type = search_type or config.SEARCH_TYPE
    threshold = config.RELEVANCE_THRESHOLD if threshold is None else threshold
    vs = vectorstore or get_vectorstore()

    # Look at a wider pool first (needed for MMR, harmless for plain similarity).
    pool_size = max(k, config.FETCH_K) if search_type == "mmr" else k
    pool = vs.similarity_search_with_relevance_scores(question, k=pool_size)
    if not pool:
        return []

    best_score = max(score for _, score in pool)
    if best_score < threshold:
        return []  # nothing in the knowledge base is close enough to the question

    if search_type == "mmr":
        # MMR picks chunks that are relevant AND different from each other (fewer duplicates).
        scores = {doc.metadata["chunk_id"]: score for doc, score in pool}
        picked = vs.max_marginal_relevance_search(question, k=k, fetch_k=pool_size)
        return [(doc, scores.get(doc.metadata["chunk_id"], 0.0)) for doc in picked]

    return pool[:k]
