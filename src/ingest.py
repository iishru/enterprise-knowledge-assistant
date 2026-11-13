# ingest.py - command line tool: load files -> split -> embed -> store in Chroma.
# Usage:  python -m src.ingest data/sample_docs          (a folder)
#         python -m src.ingest data/sample_docs/leave_policy.pdf   (one file)
#         python -m src.ingest data/sample_docs --reset  (clear the database first)
import argparse
import logging
import time
from pathlib import Path

from src.loader import load_directory, load_file
from src.splitter import chunk_stats, split_documents
from src.vectorstore import add_documents, count_chunks, get_vectorstore, list_sources, reset_vectorstore


def ingest_path(path: str | Path, reset: bool = False) -> dict:
    """Ingest a file or folder and return a summary dict."""
    path = Path(path)
    start = time.time()
    vs = get_vectorstore()
    if reset:
        reset_vectorstore(vs)

    docs = load_directory(path) if path.is_dir() else load_file(path)
    chunks = split_documents(docs)
    add_documents(chunks, vs)

    stats = chunk_stats(chunks)
    return {
        "files": len({d.metadata["source"] for d in docs}),
        "pages_or_sections": len(docs),
        "chunks_added": stats["chunks"],
        "avg_chunk_length": stats["avg_length"],
        "total_chunks_in_db": count_chunks(vs),
        "seconds": round(time.time() - start, 2),
    }


def main() -> None:
    """Command line entry point: parse arguments, ingest, print a summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Add documents to the knowledge base.")
    parser.add_argument("path", help="A file or a folder to ingest")
    parser.add_argument("--reset", action="store_true", help="Clear the database before ingesting")
    args = parser.parse_args()

    summary = ingest_path(args.path, reset=args.reset)
    print("Ingest summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("Indexed files:", ", ".join(list_sources()))


if __name__ == "__main__":
    main()
