# test_retriever.py - retrieval order, the relevance threshold, and MMR, using KeywordEmbeddings.
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.retriever import retrieve
from tests.helpers import KeywordEmbeddings

TEXTS = {
    "leave.md": "Casual leave is twelve days. Sick leave is ten days. Earned leave carries forward.",
    "leave_copy.md": "Casual leave is twelve days. Sick leave is ten days. Earned leave carries forward.",
    "security.md": "Password rules: change your password every ninety days and use the vpn for security.",
    "travel.md": "Travel hotel limits and travel claim rules for every trip.",
}


@pytest.fixture()
def store(tmp_path):
    vs = Chroma(
        collection_name="retriever_test",
        embedding_function=KeywordEmbeddings(),
        persist_directory=str(tmp_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
    docs = [Document(page_content=t, metadata={"source": s, "chunk_id": f"{s}::c1"}) for s, t in TEXTS.items()]
    vs.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    return vs


def test_relevant_chunk_comes_first_with_high_score(store):
    results = retrieve("how many sick leave days", k=3, threshold=0.3, vectorstore=store)
    assert results[0][0].metadata["source"] in {"leave.md", "leave_copy.md"}
    assert results[0][1] > 0.6
    assert all(0.0 <= score <= 1.0 for _, score in results)


def test_password_question_finds_security_doc(store):
    results = retrieve("password and vpn security", k=2, threshold=0.3, vectorstore=store)
    assert results[0][0].metadata["source"] == "security.md"


def test_off_topic_question_returns_nothing(store):
    assert retrieve("who won the football world cup", k=3, threshold=0.3, vectorstore=store) == []


def test_threshold_zero_lets_everything_through(store):
    assert len(retrieve("who won the football world cup", k=3, threshold=0.0, vectorstore=store)) == 3


def test_k_limits_number_of_results(store):
    assert len(retrieve("leave", k=1, threshold=0.0, vectorstore=store)) == 1


def test_mmr_avoids_returning_both_duplicate_chunks(store):
    similar = retrieve("casual sick leave days", k=2, search_type="similarity", threshold=0.0, vectorstore=store)
    mmr = retrieve("casual sick leave days", k=2, search_type="mmr", threshold=0.0, vectorstore=store)
    similar_sources = {d.metadata["source"] for d, _ in similar}
    mmr_sources = {d.metadata["source"] for d, _ in mmr}
    assert similar_sources == {"leave.md", "leave_copy.md"}  # plain search returns the duplicate pair
    assert len(mmr_sources & {"leave.md", "leave_copy.md"}) == 1  # MMR keeps only one of them
    assert all(isinstance(score, float) for _, score in mmr)


def test_mmr_also_respects_threshold(store):
    assert retrieve("who won the football world cup", k=2, search_type="mmr", threshold=0.3, vectorstore=store) == []
