# prompts.py - every prompt template used by the project, in one place.
from langchain_core.prompts import ChatPromptTemplate

import config

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an internal knowledge assistant for Nimbus Technologies. "
        "Answer the question using ONLY the context below. "
        f'If the answer is not in the context, reply exactly: "{config.NO_ANSWER_MESSAGE}" '
        "Keep answers short and clear. "
        "After each fact, cite its source in square brackets, copying the label shown above the "
        "passage it came from, for example [leave_policy.pdf, p.2] or [code_of_conduct.md].\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and a follow-up question, rewrite the follow-up question as a "
        "standalone question that can be understood without the history. "
        "Do NOT answer it. If it is already standalone, return it unchanged. "
        "Reply with the question only.\n\nChat history:\n{history}",
    ),
    ("human", "{question}"),
])

# Used only by the evaluation script (Phase 10): an LLM acts as a strict judge.
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict grader. Compare a generated answer with the expected answer. "
        "Reply YES only if the generated answer states the same key facts (numbers, names, "
        "limits) as the expected answer and does not contradict it. Otherwise reply NO. "
        "Reply with one word: YES or NO.",
    ),
    ("human", "Question: {question}\nExpected answer: {expected}\nGenerated answer: {generated}"),
])
