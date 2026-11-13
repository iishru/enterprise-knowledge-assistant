# api.py - REST API for the assistant (stretch goal).
# Run: uvicorn api:app --reload      then open http://127.0.0.1:8000/docs for the interactive docs.
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import config
from src.ingest import ingest_path
from src.llm import MissingAPIKeyError, get_llm
from src.rag_chain import answer_question
from src.vectorstore import count_chunks, get_vectorstore, list_sources

app = FastAPI(title="Enterprise Knowledge Assistant API", version="1.0")


class Turn(BaseModel):
    """One earlier question and the answer it received."""

    question: str
    answer: str


class AskRequest(BaseModel):
    """A question, plus the earlier conversation if it is a follow-up."""

    question: str = Field(min_length=1)
    history: list[Turn] = []


class SourceInfo(BaseModel):
    """A passage that supports the answer."""

    source: str
    page: int | None = None
    snippet: str
    score: float


class AskResponse(BaseModel):
    """The answer with its sources."""

    answer: str
    sources: list[SourceInfo]
    standalone_question: str
    latency_ms: int


@app.get("/health")
def health() -> dict:
    """Tell whether the service is up and how many chunks are indexed."""
    return {"status": "ok", "chunks": count_chunks(), "documents": list_sources()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    """Answer a question from the indexed documents."""
    vectorstore = get_vectorstore()
    if count_chunks(vectorstore) == 0:
        raise HTTPException(status_code=409, detail="The knowledge base is empty. Upload documents with /ingest first.")
    history = [(turn.question, turn.answer) for turn in request.history]
    try:
        return answer_question(request.question, history, vectorstore=vectorstore, llm=get_llm())
    except MissingAPIKeyError as error:
        raise HTTPException(status_code=503, detail=str(error))


@app.post("/ingest")
def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Add uploaded PDF, DOCX, TXT or Markdown files to the knowledge base."""
    for file in files:
        if Path(file.filename or "").suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
    with tempfile.TemporaryDirectory() as tmp:
        for file in files:
            # Keep the original file name: it becomes the "source" in citations.
            (Path(tmp) / Path(file.filename).name).write_bytes(file.file.read())
        summary = ingest_path(tmp)
    return summary
