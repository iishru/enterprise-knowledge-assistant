# test_evaluate.py - checks the metric maths, the question set itself, and the evaluation run
# (with word-overlap test embeddings and a fake LLM, so no downloads or API keys).
import re
from collections import Counter
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from eval.evaluate import (build_report, first_hit_rank, is_refusal, load_questions, mean, reciprocal_rank,
                           run_config, summary_table, sweep_table, sweep_thresholds)
from src.loader import load_directory

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


# ---------------------------------------------------------------- metric maths
def test_rank_and_reciprocal_rank():
    assert first_hit_rank(["a", "b", "c"], ["c"]) == 3
    assert first_hit_rank(["a", "b"], ["z"]) is None
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_mean_ignores_none_and_handles_empty():
    assert mean([1, 0, None, 1]) == 2 / 3
    assert mean([]) is None


def test_is_refusal():
    assert is_refusal(config.NO_ANSWER_MESSAGE)
    assert not is_refusal("Sick leave is 10 days.")


# ---------------------------------------------------------------- the 30-question file
def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_question_set_has_expected_shape():
    questions = load_questions()
    assert Counter(q["type"] for q in questions) == {"answerable": 22, "followup": 4, "unanswerable": 4}
    assert len({q["id"] for q in questions}) == 30
    for q in questions:
        assert q["question"] and q["expected_answer"]
        if q["type"] == "followup":
            assert q["history"]
        if q["type"] != "unanswerable":
            assert q["expected_sources"] and q["evidence"]


def test_every_expected_fact_really_appears_in_its_source_document():
    by_file: dict[str, str] = {}
    for doc in load_directory(SAMPLE_DIR):
        by_file[doc.metadata["source"]] = by_file.get(doc.metadata["source"], "") + " " + doc.page_content
    for q in load_questions():
        if q["type"] == "unanswerable":
            continue
        # the evidence snippet must be in at least one of the expected files
        assert any(normalise(q["evidence"]) in normalise(by_file[f]) for f in q["expected_sources"]), q["id"]


def test_unanswerable_questions_are_really_absent_from_all_documents():
    everything = normalise(" ".join(d.page_content for d in load_directory(SAMPLE_DIR))).lower()
    for q in load_questions():
        for term in q.get("absent_terms", []):
            assert not re.search(rf"\b{re.escape(term.lower())}\b", everything), (q["id"], term)  # whole word


# ---------------------------------------------------------------- running an evaluation
def fake_llm():
    """Answers with the right fact for one question, judges by substring, rewrites follow-ups."""
    def respond(prompt):
        text = prompt.to_string()
        if "strict grader" in text:
            expected = re.search(r"Expected answer: (.*)\n", text).group(1)
            generated = re.search(r"Generated answer: (.*)", text).group(1)
            return AIMessage(content="YES" if expected.rstrip(".") in generated else "NO")
        if "standalone question" in text:
            return AIMessage(content="How many sick leave days are there?")
        return AIMessage(content="10 days per year. [leave_policy.pdf, p.2]")
    return RunnableLambda(respond)


QUESTIONS = [
    {"id": "A1", "type": "answerable", "question": "How many sick leave days?", "expected_answer": "10 days per year",
     "expected_sources": ["leave_policy.pdf"]},
    {"id": "A2", "type": "answerable", "question": "What are the password and vpn security rules?",
     "expected_answer": "90 days", "expected_sources": ["it_security_policy.pdf"]},
    {"id": "F1", "type": "followup", "question": "And the other one?", "expected_answer": "10 days per year",
     "expected_sources": ["leave_policy.pdf"], "history": [["How many casual leave days?", "12 days."]]},
    {"id": "U1", "type": "unanswerable", "question": "who won the football world cup", "expected_answer": "-",
     "expected_sources": []},
]


def test_retrieval_only_run_skips_followups(keyword_store):
    run = run_config(keyword_store, QUESTIONS, k=3, search_type="similarity", threshold=0.3, llm=None)
    s = run["summary"]
    assert s["n_skipped"] == 1 and s["n_answerable"] == 2 and s["n_unanswerable"] == 1
    assert s["hit_rate"] == 1.0 and s["mrr"] == 1.0
    assert s["refusal_accuracy"] == 1.0 and s["false_refusal_rate"] == 0.0
    assert s["answer_accuracy"] is None


def test_full_run_with_fake_llm_counts_accuracy_and_refusals(keyword_store):
    run = run_config(keyword_store, QUESTIONS, k=3, search_type="similarity", threshold=0.3, llm=fake_llm())
    s = run["summary"]
    assert s["n_skipped"] == 0 and s["n_answerable"] == 3
    # A1 and F1 get "10 days per year" (correct); A2's expected "90 days" is not in that answer (wrong)
    assert abs(s["answer_accuracy"] - 2 / 3) < 1e-9
    assert s["hit_rate"] == 1.0
    assert s["refusal_accuracy"] == 1.0  # the off-topic question is refused by the threshold
    assert s["avg_latency_ms"] is not None


def test_threshold_sweep_separates_answerable_from_unanswerable(keyword_store):
    sweep = sweep_thresholds(keyword_store, QUESTIONS, [0.0, 0.3, 0.99])
    low, mid, high = sweep["table"]
    assert low["refusal_accuracy"] == 0.0           # too permissive: off-topic is not refused
    assert mid["false_refusal_rate"] == 0.0 and mid["refusal_accuracy"] == 1.0   # a good threshold
    assert high["false_refusal_rate"] == 1.0        # too strict: everything refused
    assert sweep["lowest_answerable_top_score"] > sweep["highest_unanswerable_top_score"]
    assert set(sweep["unanswerable_scores"]) == {"U1"}
    assert "do not overlap" in sweep_table(sweep)


def test_sweep_message_admits_overlap():
    overlapping = {"table": [], "lowest_answerable_top_score": 0.2, "highest_unanswerable_top_score": 0.5,
                   "unanswerable_scores": {"U1": 0.5}}
    assert "overlap, so no threshold separates" in sweep_table(overlapping)


def test_report_is_built_from_results(keyword_store):
    class Args:
        k, threshold, provider = 3, 0.3, "groq"

    run = run_config(keyword_store, QUESTIONS, 3, "similarity", 0.3, fake_llm())
    results = [{"label": "chunk_size=800, similarity", "n_chunks": 3, **run}]
    report = build_report(results, sweep_thresholds(keyword_store, QUESTIONS, [0.3]), Args, used_llm=True)
    assert "# Evaluation results" in report and "chunk_size=800, similarity" in report
    assert "Threshold sweep" in report and "| A1 |" in report
    assert "| chunk_size=800, similarity | 3 |" in summary_table(results)
