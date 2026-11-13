# config.py - all project settings in one place.
# Each value can be overridden from the .env file; the defaults below are what we normally use.
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# "huggingface" = the real model (default). "fake" = random vectors, only for offline tests.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface")
EMBEDDING_DIM = 384  # size of all-MiniLM-L6-v2 vectors (used by the fake backend)

# --- Vector database (Chroma, saved on disk) ---
CHROMA_DIR = str(ROOT / os.getenv("CHROMA_DIR", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_docs")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 4))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")  # "similarity" or "mmr"
FETCH_K = int(os.getenv("FETCH_K", 20))  # candidates looked at before MMR picks TOP_K
# If even the best chunk scores below this (0 to 1, higher = more similar), we say "no relevant info".
# 0.3 is a starting guess: tune it with `python -m eval.evaluate --sweep` (Phase 10).
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.3))
HISTORY_TURNS = 4  # how many past question/answer pairs are used to understand a follow-up
NO_ANSWER_MESSAGE = "I couldn't find this in the company documents."

# --- LLM (used from Phase 7 onwards) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "ollama"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
TEMPERATURE = 0  # 0 = consistent answers that stick to the facts

# --- Files we know how to read ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
