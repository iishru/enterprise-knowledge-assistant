# rag_chain.py - the whole question-answering pipeline, readable top to bottom.
# question -> (rewrite follow-up) -> retrieve chunks -> build context -> LLM -> answer + sources
import logging
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

import config
from src.llm import get_llm
from src.prompts import ANSWER_PROMPT, CONDENSE_PROMPT
from src.retriever import retrieve

logger = logging.getLogger(__name__)


def source_label(metadata: dict) -> str:
    """Citation label like 'leave_policy.pdf, p.2' (or just the file name when there is no page)."""
    page = metadata.get("page")
    return f"{metadata['source']}, p.{page}" if page else metadata["source"]


def format_context(results: list[tuple[Document, float]]) -> str:
    """Join the retrieved chunks into one context string; each passage starts with its label."""
    return "\n\n".join(f"[{source_label(doc.metadata)}]\n{doc.page_content}" for doc, _ in results)


def format_history(chat_history: list[tuple[str, str]]) -> str:
    """Turn (question, answer) pairs into plain text for the rewrite prompt."""
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in chat_history)


def condense_question(question: str, chat_history: list[tuple[str, str]], llm: Runnable) -> str:
    """Rewrite a follow-up like 'And for contractors?' into a standalone question."""
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"history": format_history(chat_history), "question": question}).strip()
    return rewritten or question


def answer_question(
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    vectorstore: Chroma | None = None,
    llm: Runnable | None = None,
    k: int | None = None,
    search_type: str | None = None,
    threshold: float | None = None,
) -> dict:
    """Answer a question from the knowledge base.

    chat_history is a list of (user_question, assistant_answer) pairs.
    Returns: answer, sources (shown to the user), retrieved (every chunk found, for evaluation),
    standalone_question, latency_ms.
    """
    start = time.time()
    history = (chat_history or [])[-config.HISTORY_TURNS:]  # keep only the last few turns

    # 1. Follow-up questions are rewritten so that retrieval can understand them.
    standalone = question
    if history:
        llm = llm or get_llm()
        standalone = condense_question(question, history, llm)
    logger.info("question=%r standalone=%r", question, standalone)

    # 2. Retrieve the top-k chunks (empty list = nothing relevant enough).
    results = retrieve(standalone, k=k, search_type=search_type, threshold=threshold, vectorstore=vectorstore)
    logger.info("retrieved %d chunks, scores=%s", len(results), [round(s, 3) for _, s in results])

    retrieved = [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata["source"],
            "page": doc.metadata.get("page"),
            "score": round(float(score), 4),
        }
        for doc, score in results
    ]

    # 3. Nothing relevant: refuse without calling the LLM (no cost, no hallucination).
    if not results:
        return {
            "answer": config.NO_ANSWER_MESSAGE,
            "sources": [],
            "retrieved": retrieved,
            "standalone_question": standalone,
            "latency_ms": int((time.time() - start) * 1000),
        }

    # 4-5. Put the chunks into the prompt and let the LLM answer from them only.
    llm = llm or get_llm()
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(results), "question": standalone}).strip()

    # 6. Sources the user can check. If the LLM itself said "not found", showing sources would mislead.
    sources = [
        {
            "source": doc.metadata["source"],
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:200].replace("\n", " "),
            "score": round(float(score), 3),
        }
        for doc, score in results
    ]
    if config.NO_ANSWER_MESSAGE in answer:
        sources = []

    latency_ms = int((time.time() - start) * 1000)
    logger.info("answered in %d ms", latency_ms)
    return {
        "answer": answer,
        "sources": sources,
        "retrieved": retrieved,
        "standalone_question": standalone,
        "latency_ms": latency_ms,
    }
