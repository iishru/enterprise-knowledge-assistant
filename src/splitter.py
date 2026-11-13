# splitter.py - cuts Documents into overlapping chunks and gives each chunk a unique chunk_id.
# chunk_id looks like "leave_policy.pdf::p2::c3" (file, page, chunk number on that page).
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split Documents into chunks, keeping source/page metadata and adding chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    counters: dict[tuple, int] = {}  # counts chunks per (file, page)
    for chunk in chunks:
        source = chunk.metadata["source"]
        page = chunk.metadata.get("page")
        key = (source, page)
        counters[key] = counters.get(key, 0) + 1
        number = counters[key]
        if page is None:
            chunk.metadata["chunk_id"] = f"{source}::c{number}"
        else:
            chunk.metadata["chunk_id"] = f"{source}::p{page}::c{number}"
    return chunks


def chunk_stats(chunks: list[Document]) -> dict:
    """Return the number of chunks and their average length."""
    count = len(chunks)
    avg = round(sum(len(c.page_content) for c in chunks) / count, 1) if count else 0
    return {"chunks": count, "avg_length": avg}
