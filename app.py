# app.py - Streamlit web UI: upload documents, chat with them, and see the sources of each answer.
# Run with:  streamlit run app.py
import tempfile
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

import config
from src.ingest import ingest_path
from src.llm import MissingAPIKeyError, get_llm
from src.rag_chain import answer_question
from src.vectorstore import count_chunks, get_vectorstore, list_sources, reset_vectorstore

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚")


@st.cache_resource
def load_vectorstore() -> Chroma:
    """Open the vector database once and reuse it across Streamlit re-runs."""
    return get_vectorstore()


@st.cache_resource
def load_llm(provider: str) -> BaseChatModel:
    """Build the chat model once per provider and reuse it."""
    return get_llm(provider)


# Chat state lives in session_state (it survives Streamlit re-runs).
if "messages" not in st.session_state:
    st.session_state.messages = []  # what is shown on screen
if "history" not in st.session_state:
    st.session_state.history = []  # (question, answer) pairs sent to the LLM for follow-ups

vectorstore = load_vectorstore()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Knowledge base")

    providers = ["groq", "ollama"]
    provider = st.selectbox(
        "LLM provider",
        providers,
        index=providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0,
        help="Groq = free hosted API (needs GROQ_API_KEY). Ollama = runs on your own computer.",
    )

    uploaded = st.file_uploader(
        "Upload documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Ingest uploaded files", disabled=not uploaded):
        try:
            with st.spinner("Reading, splitting and embedding..."), tempfile.TemporaryDirectory() as tmp:
                for file in uploaded:
                    # Keep the original file name: it becomes the "source" shown in citations.
                    (Path(tmp) / Path(file.name).name).write_bytes(file.getvalue())
                summary = ingest_path(tmp)
            st.success(f"Added {summary['files']} file(s) as {summary['chunks_added']} chunks.")
        except Exception as error:
            st.error(f"Could not ingest the files: {error}")

    st.subheader("Indexed documents")
    sources = list_sources(vectorstore)
    if sources:
        for name in sources:
            st.write(f"- {name}")
        st.caption(f"{count_chunks(vectorstore)} chunks in total")
    else:
        st.info("Nothing indexed yet. Upload files above, or run `python -m src.ingest data/sample_docs`.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    sure = st.checkbox("I'm sure I want to delete all documents")
    if st.button("Reset knowledge base", disabled=not sure):
        reset_vectorstore(vectorstore)
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------- main chat area
st.title("📚 Enterprise Knowledge Assistant")
st.caption("Ask questions about your company documents. Answers come only from the uploaded files.")


def show_message(message: dict) -> None:
    """Draw one chat message, with its sources expander and answer time if it has them."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for s in message["sources"]:
                    page = f", page {s['page']}" if s["page"] else ""
                    st.markdown(f"**{s['source']}**{page} - relevance {s['score']}")
                    st.caption(s["snippet"])
        if message.get("latency_ms") is not None:
            st.caption(f"Answered in {message['latency_ms'] / 1000:.1f} s")


for message in st.session_state.messages:
    show_message(message)

question = st.chat_input("Ask a question about your documents")
if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    if count_chunks(vectorstore) == 0:
        reply = {
            "role": "assistant",
            "content": "The knowledge base is empty. Please upload and ingest some documents first.",
        }
    else:
        try:
            with st.spinner("Thinking..."):
                llm = load_llm(provider)
                result = answer_question(question, st.session_state.history, vectorstore=vectorstore, llm=llm)
            st.session_state.history.append((question, result["answer"]))
            reply = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "latency_ms": result["latency_ms"],
            }
        except MissingAPIKeyError as error:
            reply = {"role": "assistant", "content": f"⚠️ {error}"}
        except Exception as error:  # e.g. no internet, Ollama not running
            reply = {"role": "assistant", "content": f"⚠️ Something went wrong: {error}"}

    st.session_state.messages.append(reply)
    show_message(reply)
