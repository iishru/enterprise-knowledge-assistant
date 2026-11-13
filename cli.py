# cli.py - quick terminal chat for testing the assistant without the web UI.
# Usage: python cli.py      (type 'quit' to exit)
import logging

from src.llm import MissingAPIKeyError
from src.rag_chain import answer_question
from src.vectorstore import count_chunks


def main() -> None:
    """Run the terminal chat loop."""
    logging.basicConfig(level=logging.WARNING)
    if count_chunks() == 0:
        print("The knowledge base is empty. Run first:  python -m src.ingest data/sample_docs")
        return

    print("Enterprise Knowledge Assistant - ask a question (or type 'quit').")
    history: list[tuple[str, str]] = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        try:
            result = answer_question(question, history)
        except MissingAPIKeyError as error:
            print(f"\n{error}")
            return
        except Exception as error:  # e.g. no internet, Ollama not running
            print(f"\nSomething went wrong: {error}")
            continue

        print(f"\nAssistant: {result['answer']}")
        for s in result["sources"]:
            page = f", p.{s['page']}" if s["page"] else ""
            print(f"   - {s['source']}{page} (score {s['score']})")
        print(f"   ({result['latency_ms']} ms)")
        history.append((question, result["answer"]))


if __name__ == "__main__":
    main()
