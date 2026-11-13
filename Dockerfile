# Dockerfile - runs the Streamlit app. Works locally and on Hugging Face Spaces (Docker SDK, port 7860).
#   docker build -t knowledge-assistant .
#   docker run -p 7860:7860 -e GROQ_API_KEY=your_key knowledge-assistant
FROM python:3.11-slim

# Hugging Face Spaces runs containers as user 1000, so we do the same.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only PyTorch first (the default build pulls a GPU version that is ~2 GB larger).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model now, so the first question is not slow.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .

# Index the sample documents so the demo works right away. Files uploaded later live only until the
# container restarts (the disk is temporary on Spaces).
RUN python -m src.ingest data/sample_docs --reset

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
