# test_rag_chain.py - tests the pipeline with a fake LLM, so no API key or internet is needed.
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from src.rag_chain import answer_question, format_context, source_label


def never_called_llm():
    def boom(_):
        raise AssertionError("The LLM must not be called in this case")
    return RunnableLambda(boom)


def test_source_label_with_and_without_page():
    assert source_label({"source": "a.pdf", "page": 2}) == "a.pdf, p.2"
    assert source_label({"source": "b.md"}) == "b.md"


def test_grounded_answer_returns_llm_text_and_sources(keyword_store):
    llm = FakeListChatModel(responses=["Sick leave is 10 days. [leave_policy.pdf, p.2]"])
    result = answer_question("how many sick leave days", vectorstore=keyword_store, llm=llm, threshold=0.3)
    assert result["answer"] == "Sick leave is 10 days. [leave_policy.pdf, p.2]"
    assert result["sources"][0]["source"] == "leave_policy.pdf"
    assert result["sources"][0]["page"] == 2
    assert len(result["sources"][0]["snippet"]) <= 200
    assert result["retrieved"][0]["chunk_id"] == "leave_policy.pdf::c1"
    assert result["latency_ms"] >= 0


def test_context_contains_labels_and_text(keyword_store):
    prompts = []
    llm = RunnableLambda(lambda p: prompts.append(p.to_string()) or AIMessage(content="ok"))
    answer_question("sick leave days", vectorstore=keyword_store, llm=llm, threshold=0.3)
    assert "[leave_policy.pdf, p.2]" in prompts[0]
    assert "Sick leave is ten days" in prompts[0]
    assert "ONLY the context" in prompts[0]


def test_off_topic_question_is_refused_without_calling_llm(keyword_store):
    result = answer_question("who won the football world cup", vectorstore=keyword_store,
                             llm=never_called_llm(), threshold=0.3)
    assert result["answer"] == config.NO_ANSWER_MESSAGE
    assert result["sources"] == [] and result["retrieved"] == []


def test_llm_refusal_hides_sources(keyword_store):
    llm = FakeListChatModel(responses=[config.NO_ANSWER_MESSAGE])
    result = answer_question("sick leave bonus", vectorstore=keyword_store, llm=llm, threshold=0.0)
    assert result["answer"] == config.NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert result["retrieved"]  # still recorded for evaluation


def test_follow_up_is_rewritten_before_retrieval(keyword_store):
    # "What about the other type?" has no keywords, so retrieval only works after the rewrite.
    llm = FakeListChatModel(responses=["How many sick leave days are there?", "10 days. [leave_policy.pdf, p.2]"])
    history = [("How many casual leave days?", "12 days. [leave_policy.pdf, p.2]")]
    result = answer_question("What about the other type?", history, vectorstore=keyword_store, llm=llm, threshold=0.3)
    assert result["standalone_question"] == "How many sick leave days are there?"
    assert result["answer"].startswith("10 days")
    assert result["sources"][0]["source"] == "leave_policy.pdf"


def test_only_last_four_turns_of_history_are_used(keyword_store):
    seen = []
    llm = RunnableLambda(lambda p: seen.append(p.to_string()) or AIMessage(content="sick leave days"))
    history = [(f"old question {i}", f"old answer {i}") for i in range(6)]
    answer_question("and then?", history, vectorstore=keyword_store, llm=llm, threshold=0.0)
    condense_prompt = seen[0]
    assert "old question 5" in condense_prompt and "old question 2" in condense_prompt
    assert "old question 1" not in condense_prompt and "old question 0" not in condense_prompt


def test_format_context_without_page(keyword_store):
    docs = keyword_store.similarity_search_with_relevance_scores("gift rules", k=1)
    assert format_context(docs).startswith("[code_of_conduct.md]\n")
