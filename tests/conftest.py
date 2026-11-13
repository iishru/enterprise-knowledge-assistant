# conftest.py - runs before the tests. Makes the project importable and uses fake embeddings,
# so tests never download a model or need internet.
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["EMBEDDING_BACKEND"] = "fake"


import pytest  # noqa: E402


@pytest.fixture()
def keyword_store(tmp_path):
    """A small Chroma store using KeywordEmbeddings (word-overlap vectors) for pipeline tests."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    from tests.helpers import KeywordEmbeddings

    vs = Chroma(
        collection_name="pipeline_test",
        embedding_function=KeywordEmbeddings(),
        persist_directory=str(tmp_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
    rows = [
        ("leave_policy.pdf", 2, "Casual leave is twelve days. Sick leave is ten days per year."),
        ("it_security_policy.pdf", 1, "Change your password every ninety days and use the vpn for security."),
        ("code_of_conduct.md", None, "Gift rules: gifts above two thousand rupees must be reported."),
    ]
    docs = []
    for source, page, text in rows:
        meta = {"source": source, "chunk_id": f"{source}::c1"}
        if page:
            meta["page"] = page
        docs.append(Document(page_content=text, metadata=meta))
    vs.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    return vs
