# loader.py - reads PDF, DOCX, TXT and Markdown files and turns them into LangChain Documents.
# Every Document gets a clean "source" (file name) and, for PDFs, a 1-based "page" number.
import logging
import re
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Collapse repeated spaces/tabs and extra blank lines, keeping paragraph breaks."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: str | Path) -> list[Document]:
    """Load one file. PDFs give one Document per page; other types give one Document."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        raw_docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        raw_docs = Docx2txtLoader(str(path)).load()
    elif ext in {".txt", ".md"}:
        raw_docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = []
    for doc in raw_docs:
        text = clean_text(doc.page_content)
        if not text:  # skip empty pages
            continue
        metadata = {"source": path.name}  # file name only, not the full path
        if ext == ".pdf":
            metadata["page"] = int(doc.metadata.get("page", 0)) + 1  # PyPDF pages start at 0
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def load_directory(folder: str | Path) -> list[Document]:
    """Load every supported file in a folder (not sub-folders). Unsupported files are skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    docs: list[Document] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file: %s", path.name)
            continue
        docs.extend(load_file(path))
    return docs
