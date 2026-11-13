# NOTES - how this project works (for a beginner)

## The idea in one paragraph

A language model (the LLM) doesn't know your company's documents. So instead of asking it directly, we first **search** the documents for the few passages that relate to the question, paste those passages into the prompt, and tell the LLM: "answer only from these". This is **RAG** (Retrieval-Augmented Generation). The search works on **embeddings**: a model turns every passage into a list of 384 numbers (a vector) so that passages with similar meaning have similar vectors. A **vector database** (ChromaDB) stores those vectors and finds the closest ones to a question quickly.

## What happens when you add a document (indexing)

`python -m src.ingest data/sample_docs` does this:

1. **loader.py** reads each file (PDF page by page, DOCX, Markdown, TXT) into LangChain `Document` objects, cleans extra spaces, and records `source` (file name) and `page`.
2. **splitter.py** cuts each document into overlapping chunks of about 800 characters, because a whole document is too big to search precisely. Each chunk gets an id like `leave_policy.pdf::p2::c3`.
3. **embeddings.py** loads the embedding model (`all-MiniLM-L6-v2`).
4. **vectorstore.py** stores every chunk, its vector and its metadata in ChromaDB on disk (`chroma_db/`). Before adding a file it deletes that file's old chunks, so running ingest twice never creates duplicates.
5. **ingest.py** ties these steps together and prints a summary.

## What happens when you ask a question

`answer_question()` in **rag_chain.py**:

1. **Follow-up?** If there is earlier conversation, the LLM rewrites "And for contractors?" into a full standalone question (prompt in **prompts.py**), because the search can't understand "and" or "that one".
2. **Retrieve** (**retriever.py**): the question is embedded and ChromaDB returns the 4 closest chunks, each with a score from 0 to 1. If the best score is below the threshold (0.3), nothing is relevant and we stop here with "I couldn't find this in the company documents." No LLM call is made.
3. **Build the prompt**: each chunk is labelled `[file, p.N]` and put into the answer prompt with the rule "use ONLY this context".
4. **Ask the LLM** (**llm.py** chooses Groq or Ollama) and read back the answer, which cites those labels.
5. **Return** the answer, the sources (file, page, snippet, score) and the time taken.

## The files

| File | What it does |
|---|---|
| `config.py` | Every setting (chunk size, top-k, threshold, model names) in one place; values can be overridden in `.env`. |
| `src/loader.py` | Files to `Document` objects. |
| `src/splitter.py` | Documents to chunks with ids. |
| `src/embeddings.py` | The embedding model (a fake one exists for offline tests). |
| `src/vectorstore.py` | Open, add to, list and reset the ChromaDB database. |
| `src/retriever.py` | Find the top-k chunks; apply the relevance threshold; optional MMR (picks relevant chunks that differ from each other). |
| `src/llm.py` | Build the chat model; clear error when the Groq key is missing. |
| `src/prompts.py` | The answer, rewrite and judge prompts. |
| `src/rag_chain.py` | The pipeline above. |
| `src/ingest.py` | Command-line indexing. |
| `app.py` | Streamlit web UI. |
| `cli.py` | Terminal chat. |
| `api.py` | REST API (`/health`, `/ask`, `/ingest`). |
| `eval/test_questions.json` | 30 questions with expected answers and sources. |
| `eval/evaluate.py` | Runs the questions and computes the metrics; `--sweep` helps pick the threshold. |
| `data/make_sample_docs.py` | Creates the 8 fictional "Nimbus Technologies" documents. |
| `tests/` | Automated tests. Most use small test doubles (word-overlap embeddings, a fake LLM) so they run without internet. |

## Words worth knowing for interviews

- **Chunk overlap**: neighbouring chunks share 150 characters so a sentence cut at a boundary still appears whole in one of them.
- **Cosine similarity**: how close two vectors point; 1 means same direction. Chroma returns a distance and we turn it into a score with `1 - distance`.
- **Hit Rate@k**: for how many questions did the right document appear in the top k chunks.
- **MRR**: average of 1 / (position of the first correct chunk); rewards putting the right chunk first.
- **LCEL**: LangChain's `prompt | llm | parser` syntax for chaining steps.
- **Hallucination**: the model inventing an answer. We reduce it with the strict prompt, temperature 0, the relevance threshold, and citations that let you check.

## Things that are easy to get wrong

- Change the embedding model, or the chunk size, and you must re-ingest (`--reset`); old vectors don't match the new ones.
- The 0.3 threshold is a starting guess. Run `python -m eval.evaluate --no-llm --sweep` with the real model and adjust.
- The evaluation's "answer accuracy" comes from an LLM judge, which can be wrong; read the per-question table yourself.
