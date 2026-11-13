# evaluate.py - measures how good the assistant is, on the 30 questions in test_questions.json.
#
# Metrics (these are the numbers that may go on the resume - only real, measured ones):
#   Hit Rate@k      share of answerable questions where the right document is in the top-k chunks
#   MRR             average of 1/rank of the first chunk from the right document
#   Answer accuracy share of answerable questions answered correctly (an LLM acts as a strict judge)
#   False refusals  share of answerable questions where the assistant wrongly said "not found"
#   Refusal acc.    share of unanswerable questions where the assistant correctly refused
#   Latency         average time per question
#
# Examples:
#   python -m eval.evaluate --no-llm --sweep                   (retrieval only, no API key; tune the threshold)
#   python -m eval.evaluate --chunk-sizes 500 800 --search-types similarity mmr --write
import argparse
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

import config
from src.llm import MissingAPIKeyError, get_llm
from src.loader import load_directory
from src.prompts import JUDGE_PROMPT
from src.rag_chain import answer_question
from src.retriever import retrieve
from src.splitter import split_documents
from src.vectorstore import add_documents, get_vectorstore

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = EVAL_DIR / "test_questions.json"
DEFAULT_DOCS = config.ROOT / "data" / "sample_docs"
RESULTS_PATH = EVAL_DIR / "results.md"
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


# ------------------------------------------------------------------ small pure helpers (unit-tested)
def load_questions(path: str | Path = DEFAULT_QUESTIONS) -> list[dict]:
    """Read the evaluation questions from a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_hit_rank(retrieved_sources: list[str], expected_sources: list[str]) -> int | None:
    """1-based position of the first retrieved chunk that comes from an expected file, else None."""
    for position, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return position
    return None


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """1/rank of the first correct chunk, or 0 if the right file was not retrieved."""
    rank = first_hit_rank(retrieved_sources, expected_sources)
    return 1.0 / rank if rank else 0.0


def is_refusal(answer: str) -> bool:
    """True if the answer is the standard 'not found in the documents' reply."""
    return config.NO_ANSWER_MESSAGE in answer


def mean(values: list) -> float | None:
    """Average of the values, ignoring None; None when there is nothing to average."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pct(value: float | None) -> str:
    """Format 0.82 as '82%' (or 'n/a' for None)."""
    return "n/a" if value is None else f"{value * 100:.0f}%"


def with_retry(fn: Callable[[], Any], tries: int = 3, wait: float = 5.0) -> Any:
    """Call fn(); retry a few times (free API tiers rate-limit), then raise the last error."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception:
            if attempt == tries:
                raise
            time.sleep(wait * attempt)


def judge_answer(llm: Runnable, question: str, expected: str, generated: str) -> bool:
    """Ask the LLM (as a strict grader) whether the generated answer matches the expected one."""
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    verdict = with_retry(lambda: chain.invoke({"question": question, "expected": expected, "generated": generated}))
    return verdict.strip().upper().startswith("YES")


# ------------------------------------------------------------------ running one configuration
def build_store(docs_dir: str | Path, chunk_size: int, tmp_dir: str | Path, name: str) -> tuple[Chroma, int]:
    """Index the documents with a given chunk size into a throw-away database (the real one is untouched)."""
    overlap = round(chunk_size * config.CHUNK_OVERLAP / config.CHUNK_SIZE)  # same overlap ratio as default
    chunks = split_documents(load_directory(docs_dir), chunk_size, overlap)
    vs = get_vectorstore(str(Path(tmp_dir) / name), f"eval_{name}")
    add_documents(chunks, vs)
    return vs, len(chunks)


def run_config(vs: Chroma, questions: list[dict], k: int, search_type: str, threshold: float,
               llm: Runnable | None = None, delay: float = 0.0) -> dict:
    """Run every question and return per-question rows plus the summary metrics.

    llm=None means retrieval-only: no answers are generated, so follow-up questions (which need the
    LLM to be rewritten) are skipped and answer accuracy is not measured.
    """
    rows = []
    for item in questions:
        kind = item["type"]
        row = {"id": item["id"], "type": kind, "question": item["question"], "skipped": False,
               "hit": None, "rr": None, "refused": None, "correct": None, "answer": None, "latency_ms": None}
        if llm is None and kind == "followup":
            row["skipped"] = True
            rows.append(row)
            continue

        history = [tuple(pair) for pair in item.get("history", [])]
        started = time.time()
        if llm is not None:
            result = with_retry(lambda: answer_question(
                item["question"], history, vectorstore=vs, llm=llm,
                k=k, search_type=search_type, threshold=threshold))
            retrieved_sources = [r["source"] for r in result["retrieved"]]
            row["answer"] = result["answer"]
            row["latency_ms"] = result["latency_ms"]
            row["refused"] = is_refusal(result["answer"])
        else:
            found = retrieve(item["question"], k=k, search_type=search_type, threshold=threshold, vectorstore=vs)
            retrieved_sources = [doc.metadata["source"] for doc, _ in found]
            row["latency_ms"] = int((time.time() - started) * 1000)
            row["refused"] = not found

        if kind != "unanswerable":
            row["rr"] = reciprocal_rank(retrieved_sources, item["expected_sources"])
            row["hit"] = row["rr"] > 0
            if llm is not None:
                row["correct"] = False if row["refused"] else judge_answer(
                    llm, item["question"], item["expected_answer"], row["answer"])
        rows.append(row)
        if delay:
            time.sleep(delay)

    done = [r for r in rows if not r["skipped"]]
    answerable = [r for r in done if r["type"] != "unanswerable"]
    unanswerable = [r for r in done if r["type"] == "unanswerable"]
    summary = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_skipped": len(rows) - len(done),
        "hit_rate": mean([r["hit"] for r in answerable]),
        "mrr": mean([r["rr"] for r in answerable]),
        "answer_accuracy": mean([r["correct"] for r in answerable]),
        "false_refusal_rate": mean([r["refused"] for r in answerable]),
        "refusal_accuracy": mean([r["refused"] for r in unanswerable]),
        "avg_latency_ms": mean([r["latency_ms"] for r in done]),
    }
    return {"rows": rows, "summary": summary}


def sweep_thresholds(vs: Chroma, questions: list[dict], thresholds: list[float]) -> dict:
    """For each threshold: how many answerable questions get wrongly refused, and how many
    unanswerable ones get correctly refused? Uses only the score of the best chunk (no LLM)."""
    top = {}
    for item in questions:
        if item["type"] == "followup":  # needs the LLM to be rewritten first
            continue
        hits = vs.similarity_search_with_relevance_scores(item["question"], k=1)
        top[item["id"]] = (item["type"], hits[0][1] if hits else 0.0)
    answerable = [s for t, s in top.values() if t == "answerable"]
    unanswerable = [s for t, s in top.values() if t == "unanswerable"]
    table = [{
        "threshold": t,
        "false_refusal_rate": mean([s < t for s in answerable]),
        "refusal_accuracy": mean([s < t for s in unanswerable]),
    } for t in thresholds]
    return {
        "table": table,
        "lowest_answerable_top_score": min(answerable) if answerable else None,
        "highest_unanswerable_top_score": max(unanswerable) if unanswerable else None,
        "unanswerable_scores": {i: s for i, (t, s) in top.items() if t == "unanswerable"},
    }


# ------------------------------------------------------------------ printing and report
def summary_table(results: list[dict]) -> str:
    """Markdown table with one row per configuration."""
    lines = ["| Configuration | Chunks | Hit Rate@k | MRR | Answer accuracy | False refusals | Refusal accuracy | Avg latency |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["summary"]
        mrr = "n/a" if s["mrr"] is None else f"{s['mrr']:.2f}"
        latency = "n/a" if s["avg_latency_ms"] is None else f"{s['avg_latency_ms']:.0f} ms"
        lines.append(f"| {r['label']} | {r['n_chunks']} | {pct(s['hit_rate'])} | {mrr} | {pct(s['answer_accuracy'])} | "
                     f"{pct(s['false_refusal_rate'])} | {pct(s['refusal_accuracy'])} | {latency} |")
    return "\n".join(lines)


def rows_table(rows: list[dict]) -> str:
    """Markdown table with one line per question."""
    lines = ["| ID | Type | Right file found | Refused | Judged correct | Question |", "|---|---|---|---|---|---|"]
    mark = lambda v: "-" if v is None else ("yes" if v else "no")  # noqa: E731
    for r in rows:
        if r["skipped"]:
            lines.append(f"| {r['id']} | {r['type']} | skipped | skipped | skipped | {r['question']} |")
        else:
            lines.append(f"| {r['id']} | {r['type']} | {mark(r['hit'])} | {mark(r['refused'])} | "
                         f"{mark(r['correct'])} | {r['question']} |")
    return "\n".join(lines)


def sweep_table(sweep: dict) -> str:
    """Markdown table showing how each threshold affects refusals."""
    lines = ["| Threshold | Answerable wrongly refused | Unanswerable correctly refused |", "|---|---|---|"]
    for row in sweep["table"]:
        lines.append(f"| {row['threshold']:.2f} | {pct(row['false_refusal_rate'])} | {pct(row['refusal_accuracy'])} |")
    low, high = sweep["lowest_answerable_top_score"], sweep["highest_unanswerable_top_score"]
    scores = ", ".join(f"{i}: {s:.2f}" for i, s in sweep["unanswerable_scores"].items())
    lines.append("")
    lines.append(f"Best-chunk score of each unanswerable question: {scores}.")
    lines.append(f"Lowest best-chunk score among answerable questions: {low:.3f}. "
                 f"Highest among unanswerable questions: {high:.3f}.")
    if low > high:
        lines.append("The two ranges do not overlap, so any threshold between them separates the questions cleanly.")
    else:
        lines.append("The ranges overlap, so no threshold separates them perfectly. Pick the highest threshold "
                     "that still keeps wrongly refused answerable questions at 0%; the LLM's own 'not found' "
                     "answer has to catch the unanswerable questions that score above it.")
    return "\n".join(lines)


def build_report(results: list[dict], sweep: dict | None, args: argparse.Namespace, used_llm: bool) -> str:
    """Build the text of eval/results.md from the run results."""
    model = (config.GROQ_MODEL if (args.provider or config.LLM_PROVIDER) == "groq" else config.OLLAMA_MODEL) if used_llm else "none (retrieval only)"
    out = [
        "# Evaluation results", "",
        f"- Date: {date.today().isoformat()}",
        f"- Embedding model: `{config.EMBEDDING_MODEL}`",
        f"- LLM: `{model}`",
        f"- Top-k: {args.k}, relevance threshold: {args.threshold}",
        "- Questions: 22 answerable, 4 follow-ups, 4 unanswerable (`eval/test_questions.json`)",
        "", "## Summary", "", summary_table(results), "",
        "How to read it: Hit Rate@k and MRR measure the search step; Answer accuracy is judged by an LLM, "
        "so check the per-question tables by hand before quoting it. False refusals are answerable questions "
        "the assistant wrongly declined. Refusal accuracy covers the unanswerable questions only.",
    ]
    if not used_llm:
        out += ["", "> Retrieval-only run: no answers were generated, so Answer accuracy is n/a and follow-up "
                    "questions were skipped."]
    if sweep:
        out += ["", "## Threshold sweep", "", sweep_table(sweep)]
    out += ["", "## Per-question results"]
    for r in results:
        out += ["", f"<details><summary>{r['label']}</summary>", "", rows_table(r["rows"]), "", "</details>"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ command line
def main() -> int:
    """Command line entry point; returns the exit code."""
    p = argparse.ArgumentParser(description="Evaluate the assistant on the test questions.")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    p.add_argument("--docs", default=str(DEFAULT_DOCS))
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=[config.CHUNK_SIZE])
    p.add_argument("--search-types", nargs="+", default=[config.SEARCH_TYPE], choices=["similarity", "mmr"])
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--threshold", type=float, default=config.RELEVANCE_THRESHOLD)
    p.add_argument("--provider", default=None, help="groq or ollama (default: LLM_PROVIDER)")
    p.add_argument("--no-llm", action="store_true", help="Retrieval metrics only (no API key needed)")
    p.add_argument("--sweep", action="store_true", help="Show how the relevance threshold affects refusals")
    p.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between questions (rate limits)")
    p.add_argument("--write", action="store_true", help="Write eval/results.md")
    args = p.parse_args()

    questions = load_questions(args.questions)
    placeholder = config.EMBEDDING_BACKEND == "fake"
    if placeholder:
        print("WARNING: EMBEDDING_BACKEND=fake - the vectors are random, so these numbers mean nothing.\n"
              "This run only checks that the script works. results.md will NOT be written.\n")

    llm = None
    if not args.no_llm:
        try:
            llm = get_llm(args.provider)
        except MissingAPIKeyError as error:
            print(f"{error}\nTip: run with --no-llm to measure retrieval only.")
            return 1

    results, sweep = [], None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for size in args.chunk_sizes:
            vs, n_chunks = build_store(args.docs, size, tmp, f"cs{size}")
            if args.sweep and sweep is None:
                sweep = sweep_thresholds(vs, questions, DEFAULT_THRESHOLDS)
            for search_type in args.search_types:
                label = f"chunk_size={size}, {search_type}"
                print(f"Running: {label} ...")
                run = run_config(vs, questions, args.k, search_type, args.threshold, llm, args.delay)
                results.append({"label": label, "n_chunks": n_chunks, **run})

    for r in results:
        print(f"\n### {r['label']}\n{rows_table(r['rows'])}")
    print("\n## Summary\n" + summary_table(results))
    if sweep:
        print("\n## Threshold sweep\n" + sweep_table(sweep))
    if args.no_llm:
        print("\nRetrieval-only run: answer accuracy is n/a and follow-up questions were skipped.")

    if args.write:
        if placeholder:
            print("\nNot writing results.md (fake embeddings).")
        else:
            RESULTS_PATH.write_text(build_report(results, sweep, args, llm is not None), encoding="utf-8")
            print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
