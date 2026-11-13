# vectorstore.py - create / open the Chroma database, add chunks, list sources, reset.
# The database is saved on disk (CHROMA_DIR), so it survives restarts.
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.embeddings import get_embeddings


def get_vectorstore(
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Open the Chroma collection (creating it if it does not exist yet)."""
    return Chroma(
        collection_name=collection_name or config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory or config.CHROMA_DIR,
        # cosine distance matches the normalized embeddings
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks: list[Document], vectorstore: Chroma | None = None) -> int:
    """Store chunks. Re-ingesting a file replaces its old chunks, so there are no duplicates."""
    if not chunks:
        return 0
    vs = vectorstore or get_vectorstore()

    # Remove the old chunks of these files first (also drops chunks that no longer exist
    # when a file got shorter), then add the new ones.
    for source in sorted({c.metadata["source"] for c in chunks}):
        vs.delete(where={"source": source})

    ids = [c.metadata["chunk_id"] for c in chunks]
    vs.add_documents(chunks, ids=ids)
    return len(chunks)


def count_chunks(vectorstore: Chroma | None = None) -> int:
    """Number of chunks currently stored."""
    vs = vectorstore or get_vectorstore()
    return vs._collection.count()


def list_sources(vectorstore: Chroma | None = None) -> list[str]:
    """Unique file names that are currently indexed."""
    vs = vectorstore or get_vectorstore()
    metadatas = vs.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metadatas if m and "source" in m})


def reset_vectorstore(vectorstore: Chroma | None = None) -> None:
    """Delete every chunk (the collection stays, but is empty)."""
    vs = vectorstore or get_vectorstore()
    vs.reset_collection()
