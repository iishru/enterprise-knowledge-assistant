# test_vectorstore.py - checks storing, no duplicates on re-ingest, updating, and persistence.
# Uses fake embeddings (set in conftest.py) and a temporary folder, so it needs no internet.
from pathlib import Path

from langchain_core.documents import Document

from src.loader import load_directory
from src.splitter import split_documents
from src.vectorstore import add_documents, count_chunks, get_vectorstore, list_sources, reset_vectorstore

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def make_chunk(source: str, n: int, text: str) -> Document:
    return Document(page_content=text, metadata={"source": source, "chunk_id": f"{source}::c{n}"})


def test_reingesting_does_not_duplicate(tmp_path):
    vs = get_vectorstore(str(tmp_path), "test_col_1")
    chunks = split_documents(load_directory(SAMPLE_DIR))
    add_documents(chunks, vs)
    first = count_chunks(vs)
    add_documents(chunks, vs)
    assert first == len(chunks)
    assert count_chunks(vs) == first


def test_reingesting_a_shorter_file_removes_old_chunks(tmp_path):
    vs = get_vectorstore(str(tmp_path), "test_col_2")
    add_documents([make_chunk("a.md", 1, "one"), make_chunk("a.md", 2, "two")], vs)
    add_documents([make_chunk("a.md", 1, "only one now")], vs)
    assert count_chunks(vs) == 1


def test_metadata_is_stored_and_sources_listed(tmp_path):
    vs = get_vectorstore(str(tmp_path), "test_col_3")
    add_documents(split_documents(load_directory(SAMPLE_DIR)), vs)
    assert len(list_sources(vs)) == 8
    stored = vs.get(include=["metadatas"])["metadatas"]
    assert all("chunk_id" in m and "source" in m for m in stored)


def test_data_persists_after_reopening(tmp_path):
    add_documents([make_chunk("p.md", 1, "persist me")], get_vectorstore(str(tmp_path), "test_col_4"))
    reopened = get_vectorstore(str(tmp_path), "test_col_4")
    assert count_chunks(reopened) == 1


def test_reset_clears_everything(tmp_path):
    vs = get_vectorstore(str(tmp_path), "test_col_5")
    add_documents([make_chunk("r.md", 1, "gone soon")], vs)
    reset_vectorstore(vs)
    assert count_chunks(vs) == 0
