"""
Terminal entry point for the University Orientation Assistant.

Run with:
    python app.py

Type "quit" (or "exit") at any time to end the session.
"""

from project.chains.memory_chain import load_memory, save_memory
from project.graph import build_graph
from project.llm import model

QUIT_WORDS = {"quit", "exit"}


def _log_question(query: str) -> None:
    """Appends every question asked to memory.json's previous_questions,
    regardless of which chain handled it."""
    memory = load_memory()
    memory.setdefault("previous_questions", []).append(query)
    save_memory(memory)


def main() -> None:
    if model is None:
        raise RuntimeError("Set `model` in project/llm.py to a real chat model instance.")

    graph = build_graph()

    print("University Orientation Assistant — type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue
        if query.lower() in QUIT_WORDS:
            print("Assistant: Goodbye!")
            break

        _log_question(query)

        result = graph.invoke({"query": query})
        print(f"Assistant: {result['answer']}\n")


if __name__ == "__main__":
    main()