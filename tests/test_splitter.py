# test_splitter.py - checks chunk sizes, metadata and unique chunk ids.
from pathlib import Path

from langchain_core.documents import Document

from src.loader import load_directory
from src.splitter import chunk_stats, split_documents

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_chunks_keep_metadata_and_get_chunk_id():
    doc = Document(page_content="word " * 400, metadata={"source": "x.pdf", "page": 2})
    chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "x.pdf" and c.metadata["page"] == 2 for c in chunks)
    assert chunks[0].metadata["chunk_id"] == "x.pdf::p2::c1"
    assert chunks[1].metadata["chunk_id"] == "x.pdf::p2::c2"


def test_chunk_id_without_page():
    doc = Document(page_content="hello world", metadata={"source": "n.md"})
    chunks = split_documents([doc])
    assert chunks[0].metadata["chunk_id"] == "n.md::c1"


def test_no_chunk_is_longer_than_chunk_size():
    doc = Document(page_content="sentence one. " * 300, metadata={"source": "s.txt"})
    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)
    assert max(len(c.page_content) for c in chunks) <= 300


def test_sample_docs_chunk_ids_are_unique():
    chunks = split_documents(load_directory(SAMPLE_DIR))
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    stats = chunk_stats(chunks)
    assert stats["chunks"] == len(chunks) > 8
